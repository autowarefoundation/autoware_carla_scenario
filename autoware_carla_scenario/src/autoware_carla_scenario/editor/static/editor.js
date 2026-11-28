/* Scenario Editor — the behaviour a server render cannot do.
 *
 * Everything that changes the scenario goes through htmx and a server render, so
 * this file is limited to four things: drawing the connector between a trigger
 * and the action it fires, moving keyboard focus around the swimlanes, mounting
 * the Lanelet2 map viewer, and placing the overview's pins over the lanelets
 * they name — which only the viewer can say the position of.
 */
(function () {
  'use strict';

  /* ---------------------------------------------------------------------
   * DAG connectors
   *
   * Two different lines, because the canvas has two different relationships to
   * show and a column position expresses neither of them:
   *
   *   fires   a trigger to the action it starts. Vertical, inside one slot.
   *   causes  an action to a condition waiting on it having completed, across
   *           tracks. Drawn from `data-caused-by`, which carries the document's
   *           own reference — never from where the two cards happen to sit, so
   *           moving a clip can neither invent nor erase a causal link.
   *
   * Both go into one SVG overlay over the canvas, so adding a condition never
   * has to reason about layout — it just re-renders and we redraw.
   * ------------------------------------------------------------------- */

  var SVG_NS = 'http://www.w3.org/2000/svg';

  /* `hue` is an arrowhead's fill and a line's stroke, so which one it paints is
   * the caller's to say — an inline fill would override the stylesheet rule
   * that keeps curves unfilled. */
  function makePath(svg, d, className, hue, filled) {
    var path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    if (className) path.setAttribute('class', className);
    if (hue) path.style[filled ? 'fill' : 'stroke'] = hue;
    return svg.appendChild(path);
  }

  function drawLinks() {
    var canvas = document.getElementById('canvas');
    if (!canvas) return;
    var scroll = canvas.querySelector('.ed-canvas-scroll');
    var svg = canvas.querySelector('.ed-links');
    if (!scroll || !svg) return;

    while (svg.firstChild) svg.removeChild(svg.firstChild);

    /* Measure first, draw second.  Appending to a live SVG invalidates layout,
     * so reading a rect after each append forced one reflow per link. */
    var origin = scroll.getBoundingClientRect();
    function toX(clientX) { return clientX - origin.left + scroll.scrollLeft; }
    function toY(clientY) { return clientY - origin.top + scroll.scrollTop; }

    var laneHues = new Map();
    function laneHue(clip) {
      // The line wears the track colour of whoever runs the causing action.
      var lane = clip.closest('.ed-lane');
      if (!lane) return '';
      if (!laneHues.has(lane)) {
        laneHues.set(lane, getComputedStyle(lane).getPropertyValue('--lane-hue').trim());
      }
      return laneHues.get(lane);
    }

    var links = [];

    scroll.querySelectorAll('.ed-trigger-wrap[data-links-to]').forEach(function (wrap) {
      var target = document.getElementById(wrap.getAttribute('data-links-to'));
      if (!target) return;
      var trigger = wrap.querySelector('.ed-trigger');
      if (!trigger) return;

      var from = trigger.getBoundingClientRect();
      var to = target.getBoundingClientRect();
      var x1 = toX(from.left + from.width / 2);
      var y1 = toY(from.top);
      var x2 = toX(to.left + to.width / 2);
      var y2 = toY(to.bottom);
      if (y1 <= y2) return; // trigger is not below its action; nothing to draw

      var mid = (y1 + y2) / 2;
      links.push(['M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + mid + ', ' +
        x2 + ' ' + mid + ', ' + x2 + ' ' + (y2 + 5), '', '', false]);
      links.push(['M ' + x2 + ' ' + y2 + ' l -4 5.5 l 8 0 z', 'head', '', false]);
    });

    scroll.querySelectorAll('.ed-cond[data-caused-by]').forEach(function (card) {
      var causes = (card.getAttribute('data-caused-by') || '').split(',');
      var cardBox = card.getBoundingClientRect();

      causes.forEach(function (actionId) {
        if (!actionId) return;
        var clip = document.getElementById('node-' + actionId);
        if (!clip) return;
        var clipBox = clip.getBoundingClientRect();
        var hue = laneHue(clip);

        var x1 = toX(clipBox.right);
        var y1 = toY(clipBox.top + clipBox.height / 2);
        var x2 = toX(cardBox.left - 6);
        var y2 = toY(cardBox.top + cardBox.height / 2);
        var bend = Math.max(28, Math.abs(x2 - x1) / 2);

        links.push(['M ' + x1 + ' ' + y1 + ' C ' + (x1 + bend) + ' ' + y1 + ', ' +
          (x2 - bend) + ' ' + y2 + ', ' + x2 + ' ' + y2, 'causes', hue, false]);
        links.push(['M ' + (x2 + 5) + ' ' + y2 + ' l -6 -4 l 0 8 z',
          'causes-head', hue, true]);
      });
    });

    svg.setAttribute('width', scroll.scrollWidth);
    svg.setAttribute('height', scroll.scrollHeight);
    links.forEach(function (link) {
      makePath(svg, link[0], link[1], link[2], link[3]);
    });
  }

  var redrawTimer = null;
  function scheduleRedraw() {
    if (redrawTimer) window.clearTimeout(redrawTimer);
    redrawTimer = window.setTimeout(drawLinks, 30);
  }

  /* ---------------------------------------------------------------------
   * Lanelet2 map viewer
   *
   * simple_lanelet2 — the project that provides this framework's `lanelet2`
   * Python API — publishes its map renderer as a wasm-backed web component.
   * Using it means the preview gets real map drawing, pan and zoom, hover
   * labels and picking without any of that being reimplemented here; the
   * server keeps doing the one thing only it can, which is evaluating the
   * constraints with the real sweeper.
   *
   * It is the only renderer: there is no SVG fallback beneath it. If the module
   * cannot be loaded — no network, a locked-down machine — the preview says so
   * and the match count, which the server computes, is unaffected.
   * ------------------------------------------------------------------- */

  var viewerModule = null;   // resolved module, or false once loading has failed
  var mounted = [];          // frames holding a live viewer
  var kept = {};             // data-viewer-key -> the frame parked between renders

  /* Is this frame parked, waiting to be put back by `reuseMap`? */
  function isKept(frame) {
    var key = frame.dataset.viewerKey;
    return !!key && kept[key] === frame;
  }

  /* Destroy the viewers whose frame has left the page for good.
   *
   * htmx replaces `#editor-body` wholesale on every edit, and closing a picker
   * removes its modal, so a frame is detached without anyone telling the
   * viewer. Each one holds a parsed wasm scene; nothing was calling `destroy`,
   * so every inspector render leaked one for the life of the page.
   *
   * A parked frame is exempt. It is detached at exactly this moment -- between
   * the swap that took it off the page and the one that puts it back -- and
   * destroying it there is what would force the re-parse `reuseMap` exists to
   * avoid. */
  function reapViewers() {
    mounted = mounted.filter(function (frame) {
      if (document.contains(frame) || isKept(frame)) return true;
      try {
        if (frame.__viewer) frame.__viewer.destroy();
      } catch (error) {
        console.warn('Lanelet2 viewer would not close down:', error);
      }
      frame.__viewer = null;
      return false;
    });
  }

  function loadViewerModule(url) {
    if (viewerModule === false) return Promise.resolve(null);
    if (viewerModule) return viewerModule;
    viewerModule = import(/* webpackIgnore: true */ url).catch(function (error) {
      console.warn('Lanelet2 map viewer unavailable:', error);
      viewerModule = false;
      return null;
    });
    return viewerModule;
  }

  /* What belongs to one map: its "could not be loaded" notice, its pins.
     Scoped by `data-viewer-scope`, which every template that mounts a viewer
     marks its container with -- there is more than one map on the page now, and
     a document-wide lookup would reveal the picker's notice for the overview's
     failure. Resolved when it is needed rather than captured at mount: under
     reuse the frame outlives several renders of the fragment around it. */
  function previewOf(frame) {
    return frame.closest('[data-viewer-scope]') || document;
  }

  /* One list, because `setHighlight` is one outline colour. The server decides
     what it means -- the matches under a constraint search, the pinned lanelet
     under a fixed spawn -- so the drawing and the caption below it cannot
     disagree. Always applied, empty included: a reused viewer still shows the
     previous entity's outline until it is told otherwise. */
  function applyHighlight(frame) {
    if (frame.__viewer) frame.__viewer.setHighlight(ids(frame.dataset.highlight));
  }

  /* Put the live map back rather than building a second one.
   *
   * htmx replaces `#editor-body` on every edit and the preview then re-renders
   * itself, so the server sends a brand new, empty frame each time. Mounting
   * that frame refetches the whole .osm and parses it again in wasm -- measured
   * at one fetch per edit, for a map that has not changed. Instead the frame
   * still holding the parsed scene is swapped back in over the fresh one, and
   * only what actually differs is copied across: which entity the preview is
   * for, and which lanelets are outlined.
   *
   * The whole frame moves, not the canvas inside it. The viewer keeps a
   * reference to the element it was constructed with and observes it for
   * resizes, so lifting the canvas out from under it would leave it measuring a
   * node no longer on the page.
   *
   * Only frames the template marks with a `data-viewer-key` take part, and only
   * for the same map. Every frame has one (`map_frame` in fields.html): the
   * scenario map's, and each picker's, which carries the parsed map across the
   * body swap an edit in the side panel causes while the picker is open.
   *
   * Panning and zooming survive an edit as a consequence, which re-mounting had
   * been silently throwing away. */
  function reuseMap(fresh) {
    var key = fresh.dataset.viewerKey;
    var live = key ? kept[key] : null;
    if (!live || live === fresh) return false;
    if (live.dataset.mapSrc !== fresh.dataset.mapSrc) return false;

    fresh.replaceWith(live);
    live.dataset.entity = fresh.dataset.entity || '';
    live.dataset.highlight = fresh.dataset.highlight || '';
    if (live.__loaded) applyHighlight(live);
    return true;
  }

  function mountMap(frame) {
    if (frame.dataset.mounted === '1') return;
    var url = frame.dataset.mapViewer;
    if (!url) return;
    frame.dataset.mounted = '1';

    // Scoped now, while this fragment is certainly in the DOM: an async callback
    // looking it up later can run against a fragment htmx has already replaced.
    var unavailable = previewOf(frame).querySelector('[data-viewer-unavailable]');

    loadViewerModule(url).then(function (module) {
      if (!module || !module.LaneletViewer) {
        if (unavailable) unavailable.hidden = false;
        return;
      }

      // Laid out but not shown: the viewer needs a sized box to mount into,
      // and an empty one is what a map that never arrives would leave behind.
      frame.hidden = false;
      frame.classList.add('is-mounting');

      var viewer = new module.LaneletViewer(frame, {
        theme: 'light',
        background: 'transparent',
        scalebar: true,
      });
      frame.__viewer = viewer;
      mounted.push(frame);
      if (frame.dataset.viewerKey) kept[frame.dataset.viewerKey] = frame;

      viewer.addEventListener('load', function () {
        frame.classList.remove('is-mounting');
        frame.__loaded = true;

        // The fitted overview is the useful view here: highlighting is what
        // shows where the matches are, and zooming to the current spawn would
        // throw away the very thing the preview is for.
        applyHighlight(frame);
        layoutPins(frame);
        // Framed on the places, not on the whole city: this panel is here to
        // show where the scenario happens, and a fitted map of Nishishinjuku is
        // mostly not that. Only on load -- a pan or a zoom afterwards belongs to
        // the person doing it, and a re-render carries it across.
        frameOnPins(frame);
      });

      // Pins are drawn in page coordinates over a map that pans and zooms under
      // them, so every move is a move of theirs too.
      viewer.addEventListener('viewchange', function () {
        if (!probing) schedulePinLayout();
      });

      // Tilted, `getView` reports drawing coordinates rather than the map's --
      // a point on a hill and the ground behind it are drawn in the same place
      // -- so a pin cannot be positioned from it and is not shown at all.
      viewer.addEventListener('view3dchange', function (event) {
        frame.__tilted = !!(event.detail && event.detail.enabled);
        layoutPins(frame);
      });

      // Picking on the map is how a Lanelet2 id is chosen at all, and it writes
      // into the field the form already submits — never a second code path
      // that could save something different.
      viewer.addEventListener('select', function (event) {
        var detail = event.detail || {};
        var picked = detail.id;
        if (!picked) return;

        // A Lanelet2 map draws more than lanelets, and the layers overlap: a
        // click on a road usually lands on the direction arrow, sometimes on
        // the fill, and a hair to the side lands on a `bound` — which reports
        // the id of a *linestring*, not of the lanelet it borders. Refused
        // before the pick is routed anywhere, so both the field picker and the
        // spawn preview are covered: a boundary id must never be saved as a
        // lanelet id by either.
        var want = (frame.dataset.picksLayer || '').split(',');
        if (detail.layer && want.indexOf(detail.layer) < 0) {
          say(frame, 'That is a ' + detail.layer.replace('_', ' ') +
            ', not a lanelet. Click the lane itself.');
          return;
        }

        // A map with nothing to pick into is the places panel's: it draws what
        // the document already names, so the one thing a click there can mean
        // is "show me what named this". A place is still edited where it is
        // written -- in that object's own inspector -- so nothing is saved here.
        var input = document.getElementById(frame.dataset.picksInto || '');
        if (!input) {
          openPlace(picked);
          return;
        }

        if (frame.dataset.picksMany) {
          // Toggling, so a set is built by clicking rather than by typing a
          // comma-separated list nobody can check by eye.
          var chosen = ids(input.value);
          var at = chosen.indexOf(Number(picked));
          if (at >= 0) chosen.splice(at, 1);
          else chosen.push(Number(picked));
          input.value = chosen.join(', ');
          viewer.setHighlight(chosen);
          say(frame, chosen.length + ' selected');
          return;
        }

        input.value = String(picked);
        closePickers();
        // Dispatched last: the form's `change` trigger re-renders the whole
        // inspector, taking this modal with it.
        input.dispatchEvent(new Event('change', { bubbles: true }));
      });

      viewer.loadUrl(frame.dataset.mapSrc);
    });
  }

  function mountMaps() {
    document.querySelectorAll('.ed-map-frame[data-map-viewer]').forEach(function (frame) {
      // A picker inside a closed modal is not on screen and must not pay for a
      // wasm parse of the whole map until someone asks to see it.
      if (frame.closest('.ed-modal[hidden]')) return;
      if (reuseMap(frame)) return;
      mountMap(frame);
    });
  }

  /* ---------------------------------------------------------------------
   * Scenario pins
   *
   * The overview above the timeline draws every place a scenario names, and one
   * outline colour cannot say which of them is the spawn, which the goal and
   * which a lanelet some condition watches. The labels are the server's,
   * rendered beside the map; this is what puts each one over the lanelet it
   * names.
   *
   * Where that lanelet is on screen is the viewer's own answer, asked for
   * through its public API: `focusOn` centres the view on a primitive, so
   * focusing one and reading `getView` back reports its centre in map
   * coordinates, and the view is then put back where it was. The alternative
   * was projecting the .osm a second time here, which is a second answer to
   * "where is this" that can disagree with the drawing it is laid over.
   *
   * Each lanelet is probed once and the answer kept on the frame -- the same
   * frame that is parked across a re-render -- so an edit re-lays the pins out
   * without asking the map anything.
   * ------------------------------------------------------------------- */

  var probing = false;   // a probe moves the view; ignore its own viewchanges

  function pinLayer(frame) {
    return previewOf(frame).querySelector('[data-pins]');
  }

  /* Measure the lanelets not measured yet, in one save/restore of the view. */
  function probeCentres(frame, wanted) {
    var viewer = frame.__viewer;
    var known = frame.__pinAt || (frame.__pinAt = {});
    var missing = wanted.filter(function (id) { return !(id in known); });
    if (!viewer || !missing.length) return;

    probing = true;
    var saved = viewer.getView();
    missing.forEach(function (id) {
      // `null` records a lanelet this map does not have, so it is asked once
      // rather than on every pan.
      known[id] = null;
      if (viewer.focusOn(id)) {
        var at = viewer.getView();
        known[id] = { x: at.x, y: at.y };
      }
    });
    viewer.setView(saved);
    probing = false;
  }

  function layoutPins(frame) {
    var layer = pinLayer(frame);
    if (!layer) return;
    var viewer = frame.__viewer;
    if (!viewer || !frame.__loaded || frame.__tilted) {
      layer.hidden = true;
      return;
    }

    var pins = Array.prototype.slice.call(layer.querySelectorAll('[data-lanelet]'));
    probeCentres(frame, pins.map(function (pin) { return pin.dataset.lanelet; }));

    // Read everything first, write afterwards: this runs on every frame of a
    // pan, and a layout read taken after a style write re-flows the page. The
    // border box is what the viewer sizes itself from and reports `getView`
    // against; a pin's width is fixed for the life of a render -- the flip
    // changes neither its text nor its box -- so it is measured once and kept
    // on the element the server just made.
    var box = frame.getBoundingClientRect();
    var view = viewer.getView();
    var width = box.width;
    var height = box.height;
    pins.forEach(function (pin) {
      if (pin.__width === undefined) pin.__width = pin.offsetWidth;
    });

    var stacked = {};
    pins.forEach(function (pin) {
      var at = frame.__pinAt[pin.dataset.lanelet];
      if (!at) {
        pin.hidden = true;
        return;
      }
      // Two places on one lanelet -- an ego watching the lane it starts on --
      // would otherwise draw one pin exactly over the other.
      var rank = stacked[pin.dataset.lanelet] || 0;
      stacked[pin.dataset.lanelet] = rank + 1;
      var x = width / 2 + (at.x - view.x) * view.scale;
      var y = height / 2 - (at.y - view.y) * view.scale + rank * 19;
      // Labels on the right half read back towards the middle -- the map's
      // right edge is also where the viewer keeps its own zoom buttons -- and
      // one that would overflow anyway is flipped wherever it sits.
      var flipped = x > width * 0.5 || x + pin.__width > width - 6;
      pin.hidden = false;
      pin.classList.toggle('is-flipped', flipped);
      pin.style.transform = 'translate(' + Math.round(x) + 'px, ' + Math.round(y) +
        'px)' + (flipped ? ' translateX(-100%)' : '');
    });
    layer.hidden = false;
  }

  /* Move the view to hold every place this scenario names, once. */
  function frameOnPins(frame) {
    var viewer = frame.__viewer;
    if (!viewer || !frame.__pinAt) return;

    // The centres `layoutPins` just measured, rather than a second walk of the
    // DOM to arrive at the same numbers.
    var placed = Object.keys(frame.__pinAt)
      .map(function (id) { return frame.__pinAt[id]; })
      .filter(Boolean);
    if (!placed.length) return;
    var xs = placed.map(function (at) { return at.x; });
    var ys = placed.map(function (at) { return at.y; });

    var minX = Math.min.apply(null, xs);
    var maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys);
    var maxY = Math.max.apply(null, ys);
    var box = frame.getBoundingClientRect();
    // A floor on the span, so one place -- or two on the same street -- is
    // framed as a neighbourhood rather than as a kerbstone; and a margin, so
    // the labels have somewhere to go.
    var scale = Math.min(
      box.width / Math.max(maxX - minX, 150),
      box.height / Math.max(maxY - minY, 150)
    ) * 0.72;
    viewer.setView({ x: (minX + maxX) / 2, y: (minY + maxY) / 2, scale: scale });
  }

  function layoutEveryPin() {
    document.querySelectorAll('.ed-map-frame').forEach(layoutPins);
  }

  /* Coalesced to one layout per animation frame: the viewer emits several
   * `viewchange` events per frame during an inertial pan, and dragging a window
   * edge fires `resize` faster than the page can be laid out. */
  var pinFrame = null;
  function schedulePinLayout() {
    if (pinFrame) return;
    pinFrame = window.requestAnimationFrame(function () {
      pinFrame = null;
      layoutEveryPin();
    });
  }

  /* Open the inspector on whatever named a lanelet the places panel drew.
   *
   * The panel already lists every place as a row carrying its lanelet and the
   * request that selects its object, in document order. Reading the answer off
   * the row is why the server sends no second copy of the mapping: two lists of
   * the same thing are two lists to keep in step. */
  function openPlace(lanelet) {
    var row = placeRow(lanelet);
    var url = row && row.getAttribute('hx-get');
    if (!url || !window.htmx) return;
    window.htmx.ajax('GET', url, { target: '#inspector', swap: 'innerHTML' });
  }

  /* The panel's row for a lanelet -- the first, when two places share one. */
  function placeRow(lanelet) {
    var panel = document.getElementById('scenario-map');
    if (!panel || !(lanelet > 0)) return null;
    return panel.querySelector('[data-focus-lanelet="' + Number(lanelet) + '"]');
  }

  /* ---------------------------------------------------------------------
   * Lanelet picker
   *
   * The map opens over the viewport rather than under the field: the
   * inspector column is 384px wide, and a city at that size cannot be picked
   * from. Opening one mounts its viewer, which is the first time the map is
   * fetched and parsed at all.
   * ------------------------------------------------------------------- */

  /* The picker is moved to <body> to be shown, and removed again when it is
   * closed. It cannot simply be given a large z-index where it sits: the
   * inspector lives inside a `position: sticky` wrapper, and sticky creates a
   * stacking context whatever its own z-index is, so any z-index inside it
   * only ranks against its siblings — the canvas lanes (z-index 2) still paint
   * over it. Reparenting is what puts it in the root stacking context.
   *
   * Removing rather than hiding keeps ids unique: the next inspector render
   * builds a fresh one, so a leftover copy would be a second element with the
   * same id. */
  /* Ids as written in a field: a comma-separated list, empty tolerated. */
  function ids(text) {
    return (text || '')
      .split(',')
      .map(function (v) { return parseInt(v, 10); })
      .filter(function (v) { return !isNaN(v); });
  }

  /* Replaces the picker's subtitle. The map is the whole window while it is
   * open, so this line is the only place a refused click can be explained. */
  function say(frame, text) {
    var modal = frame.closest('.ed-modal');
    var status = modal && modal.querySelector('[data-picker-status]');
    if (status) status.textContent = text;
  }

  var openPicker = null;     // id of the picker currently portalled to <body>

  /* Ask the server for the match count of the picker that just opened.
   *
   * The panel cannot fetch it on `load`: every lanelet field on the inspector
   * has a picker in the page at once, so each render would count matches for
   * maps nobody is looking at — and counting means parsing a city. It is the
   * same reason the map viewer itself is only mounted for an open picker. */
  function primePreview(modal) {
    if (!modal || !window.htmx) return;
    modal.querySelectorAll('[hx-trigger~="picker-open"]').forEach(function (el) {
      window.htmx.trigger(el, 'picker-open');
    });
  }

  function closePickers() {
    openPicker = null;
    document.querySelectorAll('[data-portalled]').forEach(function (modal) {
      modal.remove();
    });
    reapViewers();
  }

  /* Put the picker back after an edit made inside it.
   *
   * The side panel's controls -- the spawn's fixed/searched choice, its
   * constraints -- post like every other control and swap `#editor-body`, which
   * builds a fresh copy of the modal in the inspector while the open one hangs
   * off <body>, now stale. Swapping the fresh one in keeps the panel in step
   * with the document without the map closing under the person using it: the
   * viewer itself is carried across by `reuseMap`, because the picker's frame
   * is keyed.
   *
   * Runs after `reapViewers`, so the frame in the stale copy is still on the
   * page when the reap decides what to destroy, and before `mountMaps`, which is
   * what puts the live frame into the fresh copy. */
  function reopenPicker() {
    if (!openPicker) return;
    var fresh = null;
    var stale = null;
    document.querySelectorAll('#' + CSS.escape(openPicker)).forEach(function (el) {
      if (el.dataset.portalled) stale = el;
      else fresh = el;
    });
    if (!fresh) return;
    if (stale) stale.remove();
    fresh.dataset.portalled = '1';
    document.body.appendChild(fresh);
    fresh.hidden = false;
    // The fresh copy carries an empty readout: the edit that brought it here
    // may well have been the constraint whose matches it counts.
    primePreview(fresh);
  }

  /* The matches the server just counted, drawn on the map already open beside
     them: the panel carries the ids, the frame is the one thing that can show
     them. */
  function syncPickerHighlight() {
    var panel = document.querySelector('[data-portalled] [data-picker-highlight]');
    if (!panel) return;
    var frame = document.querySelector('[data-portalled] .ed-map-frame');
    if (!frame) return;
    frame.dataset.highlight = panel.dataset.pickerHighlight || '';
    applyHighlight(frame);
  }

  /* A set is saved when the picker is closed, not on every click: sending the
   * form on each toggle would re-render the inspector and tear the map down
   * mid-selection. */
  function commitAndClose() {
    var pending = [];
    document.querySelectorAll('[data-portalled] [data-picks-many]')
      .forEach(function (frame) {
        var input = document.getElementById(frame.dataset.picksInto);
        if (input) pending.push(input);
      });
    closePickers();
    pending.forEach(function (input) {
      input.dispatchEvent(new Event('change', { bubbles: true }));
    });
  }

  document.addEventListener('click', function (event) {
    var opener = event.target.closest && event.target.closest('[data-open-picker]');
    if (opener) {
      var pickerId = opener.getAttribute('data-open-picker');
      var modal = document.getElementById(pickerId);
      if (modal) {
        closePickers();
        openPicker = pickerId;
        modal.dataset.portalled = '1';
        document.body.appendChild(modal);
        modal.hidden = false;
        mountMaps();
        primePreview(modal);
      }
      return;
    }
    // A place in the overview's list points the map at itself: the panel draws
    // a whole city, and 12 m of lane in it is not findable by eye. The row is
    // also an htmx button opening the object's inspector, so this does not
    // swallow the click -- pointing at the place and saying what named it are
    // two halves of the same answer.
    var focus = event.target.closest && event.target.closest('[data-focus-lanelet]');
    if (focus) {
      var wanted = parseInt(focus.getAttribute('data-focus-lanelet'), 10);
      var panel = document.querySelector('#scenario-map .ed-map-frame');
      if (panel && panel.__viewer && wanted > 0) {
        panel.__viewer.focusOn(wanted, { fraction: 0.3 });
      }
    }
    // Emptying a picked value: the map can set a lanelet but not unset one, and
    // a field that may be left blank -- the goal of an ego the TrafficManager
    // drives -- needs a way back to blank. It goes through the same `change` the
    // picker dispatches, so clearing and picking reach the server by one path.
    var clearer = event.target.closest && event.target.closest('[data-clear-field]');
    if (clearer) {
      var field = document.getElementById(clearer.getAttribute('data-clear-field'));
      if (field) {
        field.value = '';
        field.dispatchEvent(new Event('change', { bubbles: true }));
      }
      return;
    }
    // The scrim is the modal element itself; a click that lands on it rather
    // than on the panel inside is a click outside.
    var closer = event.target.closest && event.target.closest('[data-close-picker]');
    var scrim = event.target.classList &&
      event.target.classList.contains('ed-modal');
    if (closer || scrim) commitAndClose();
  });

  /* ---------------------------------------------------------------------
   * Selection and keyboard navigation
   *
   * Cards carry data-object-id and their own hx-get, so selecting is just
   * "activate the element". Arrow keys walk the selectable cards in document
   * order, which follows the swimlanes left to right, top to bottom.
   *
   * What takes focus is not always the element that carries the id: a
   * composite condition's wrapper holds data-object-id (it is what the
   * selection highlight outlines), but the focusable element is its head.
   * selectable() returns the focusable element of each object, so focus()
   * always moves.
   * ------------------------------------------------------------------- */

  var SELECTABLE = '#canvas [data-object-id][tabindex], ' +
    '#canvas [data-object-id] > .ed-cond-group-head[tabindex]';

  function selectable() {
    return Array.prototype.slice.call(document.querySelectorAll(SELECTABLE));
  }

  // The object an element belongs to: the nearest [data-object-id] inside the
  // canvas, which for a composite's head is its wrapper.
  function owner(element) {
    var found = element && element.closest && element.closest('[data-object-id]');
    return found && found.closest('#canvas') ? found : null;
  }

  // Moves focus by `delta` cards and reports whether it moved. Only focus that
  // is on a card, or on something inside one, navigates: anywhere else (a
  // toolbar button, the inspector, a page with no canvas) the arrow keys keep
  // their usual meaning.
  function focusOffset(delta) {
    var current = owner(document.activeElement);
    if (!current) return false;
    var nodes = selectable();
    var index = -1;
    for (var i = 0; i < nodes.length; i++) {
      if (owner(nodes[i]) === current) { index = i; break; }
    }
    if (index < 0) return false;
    var next = nodes[Math.max(0, Math.min(nodes.length - 1, index + delta))];
    if (!next || next === document.activeElement) return false;
    next.focus();
    if (document.activeElement !== next) return false;
    if (window.htmx) window.htmx.trigger(next, 'click');
    return true;
  }

  document.addEventListener('keydown', function (event) {
    if (event.target && /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;

    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      if (focusOffset(1)) event.preventDefault();
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      if (focusOffset(-1)) event.preventDefault();
    } else if (event.key === 'Enter' || event.key === ' ') {
      var active = document.activeElement;
      if (active && selectable().indexOf(active) >= 0) {
        if (window.htmx) window.htmx.trigger(active, 'click');
        event.preventDefault();
      }
    } else if (event.key === 'Escape') {
      commitAndClose();
      var panel = document.getElementById('export-panel');
      if (panel) panel.classList.add('hidden');
    }
  });

  /* --------------------------------------------------------------------- */

  window.toggleExportPanel = function () {
    var panel = document.getElementById('export-panel');
    if (panel) panel.classList.toggle('hidden');
  };

  /* ---------------------------------------------------------------------
   * Edits across a swap (#2345)
   *
   * Every inspector form posts on `change delay:200ms` and gets the whole of
   * `#editor-body` back. Two things were lost when that answer landed:
   *
   *   typing  the field the author had moved on to and was typing in is
   *           replaced by its server-rendered copy mid-word;
   *   changes a change made in the 200 ms before the swap is never posted:
   *           htmx sends nothing for an element that has left the page, and
   *           the delayed (or queued) post belongs to the replaced form.
   *
   * So just before a body swap, the field with focus and every field changed
   * but not yet posted are remembered, and found again after it by the URL
   * they post to and their name. The focused one gets focus and its caret
   * back, and what was typed -- unless the request being answered already
   * carried exactly that value, when the server's rendering of it (normalised,
   * or refused) is the one to show. Typing put back this way is posted when
   * the field is left, even if nothing more is typed; a pending change is
   * posted again at once, from the new form, which carries the new revision.
   * ------------------------------------------------------------------- */

  // Names this page to the server: see `check_revision` in editor/service.py.
  // A form one revision behind is still current when every save since was
  // this page's own -- its post in flight while the last answer was swapped.
  var CLIENT_ID = (window.crypto && window.crypto.randomUUID)
    ? window.crypto.randomUUID()
    : Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);

  // htmx wires up swapped-in content when the swap settles, 20 ms after it
  // by default, so a change made in between went to a form nothing listened
  // to yet. Nothing here animates the settle, so it happens at once.
  if (window.htmx) window.htmx.config.defaultSettleDelay = 0;

  var TOGGLES = /^(checkbox|radio)$/;
  var TYPED = /^(text|search|url|tel|email|password|number)$/;

  // A field whose value a form posts. A read-only one is set by script -- a
  // lanelet readout the map picker fills in and sends a `change` for -- so
  // its change can be dropped by a swap like a typed one.
  function isPosted(el) {
    return !!el && !!el.name && /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName) &&
      el.type !== 'hidden';
  }

  // A field the author types in or chooses from, whose focus is carried.
  function isField(el) {
    return isPosted(el) && !el.readOnly;
  }

  function isTyped(el) {
    return el.tagName === 'TEXTAREA' || (el.tagName === 'INPUT' && TYPED.test(el.type));
  }

  // The URL a field's value is posted to: its form's, or the nearest element
  // that posts on its own (a radio group with its own hx-post).
  function postOf(el) {
    var owner = ownerOf(el);
    return owner ? owner.getAttribute('hx-post') : '';
  }

  // The element that posts a field's value: its form, or a radio group.
  function ownerOf(el) {
    return el.form && el.form.getAttribute('hx-post') ? el.form : el.closest('[hx-post]');
  }

  function fieldsLike(key) {
    return Array.prototype.filter.call(
      document.querySelectorAll('input, select, textarea'),
      function (f) {
        return f.name === key.name && postOf(f) === key.post &&
          (key.value === null || f.value === key.value);
      });
  }

  function stateOf(el) {
    return TOGGLES.test(el.type) ? (el.checked ? 'on' : '') : el.value;
  }

  function setState(el, state) {
    if (TOGGLES.test(el.type)) el.checked = state === 'on';
    else el.value = state;
  }

  // Whether *state* can be put back into *el*. A <select> shows only one of
  // its options: given a value the new render no longer offers (a removed
  // actor, a phase now disabled) it would read empty, and post that (#2348).
  function canHold(el, state) {
    if (el.tagName !== 'SELECT') return true;
    for (var i = 0; i < el.options.length; i++) {
      if (el.options[i].value === state && !el.options[i].disabled) return true;
    }
    return false;
  }

  /* A form that posts to a URL holding its row's position (a signal group's,
   * a phase's, a waypoint condition's) names that row too, in hidden inputs
   * marked `data-row`: what the row, and the row it belongs to, were called
   * when rendered (see `_check_row` in editor/routes.py). After a row before
   * it was removed, or the rows reordered, the same URL is another row's, and
   * what was typed must not be put back, let alone posted, there (#2348). */
  function rowInputs(form) {
    return form
      ? Array.prototype.slice.call(form.querySelectorAll('input[type="hidden"][data-row]'))
      : [];
  }

  function formPosting(url) {
    var forms = document.querySelectorAll('form[hx-post]');
    for (var i = 0; i < forms.length; i++) {
      if (forms[i].getAttribute('hx-post') === url) return forms[i];
    }
    return null;
  }

  // What the row a `data-row` input names may be called in the next render:
  // what it was rendered as, what the request being answered renamed it to,
  // or what the field that names it says now (a rename still on its way).
  // That field is in the same form (`data-row-field`), or names the row in
  // another one (`data-row-form`, the URL of the form of the row it is in).
  function rowNames(input, config) {
    var url = input.getAttribute('data-row-form');
    if (url) {
      var owner = formPosting(url);
      var own = owner && owner.querySelector('input[type="hidden"][data-row-field]');
      return own ? rowNames(own, config) : [input.value];
    }
    var names = [input.value];
    var field = input.getAttribute('data-row-field');
    var form = input.form;
    if (!field || !form) return names;
    if (config && config.path === form.getAttribute('hx-post') &&
        config.formData && config.formData.getAll) {
      config.formData.getAll(field).forEach(function (v) { names.push(String(v).trim()); });
    }
    var named = form.elements.namedItem(field);
    if (named && typeof named.value === 'string') names.push(named.value.trim());
    return names;
  }

  function rowsOf(el, config) {
    var inputs = rowInputs(el.form);
    if (!inputs.length) return null;
    var rows = {};
    inputs.forEach(function (input) { rows[input.name] = rowNames(input, config); });
    return rows;
  }

  // Whether *el* edits the row *snap* was taken in.
  function sameRow(snap, el) {
    if (!snap.rows) return true;
    return rowInputs(el.form).every(function (input) {
      var names = snap.rows[input.name];
      return !names || names.indexOf(input.value) >= 0;
    });
  }

  // What the page last rendered into the field: the server's value.
  function renderedOf(el) {
    if (TOGGLES.test(el.type)) return el.defaultChecked ? 'on' : '';
    if (el.tagName === 'SELECT') {
      for (var i = 0; i < el.options.length; i++) {
        if (el.options[i].defaultSelected) return el.options[i].value;
      }
      return el.options.length ? el.options[0].value : '';
    }
    return el.defaultValue;
  }

  // Whether the request being answered posted the field as it is now.
  function wasPosted(snap) {
    if (!snap.posted) return false;
    if (TOGGLES.test(snap.el.type)) {
      return (snap.posted.indexOf(snap.el.value) >= 0) === (snap.state === 'on');
    }
    return snap.posted.indexOf(snap.state) >= 0;
  }

  function capture(snap) {
    var el = snap.el;
    snap.state = stateOf(el);
    snap.rendered = renderedOf(el);
    snap.start = snap.end = null;
    try {
      snap.start = el.selectionStart;
      snap.end = el.selectionEnd;
    } catch (e) { /* no selection on this kind of input */ }
    return snap;
  }

  function snapshot(el, config) {
    var key = { post: postOf(el), name: el.name, value: TOGGLES.test(el.type) ? el.value : null };
    var data = config.path === key.post && config.formData && config.formData.getAll
      ? config.formData : null;
    return capture({
      el: el,
      key: key,
      index: Math.max(0, fieldsLike(key).indexOf(el)),
      rows: rowsOf(el, config),
      posted: data ? data.getAll(el.name).map(String) : null,
    });
  }

  function select(el, start, end) {
    if (start == null || !isTyped(el)) return;
    try { el.setSelectionRange(start, end); } catch (e) { /* not selectable */ }
  }

  function post(el) {
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  // Typing put back after a swap is not on the server: post it when the field
  // is left, unless the browser does (it does when more was typed).
  function postOnLeave(el) {
    var posted = false;
    el.addEventListener('change', function () { posted = true; }, { once: true });
    el.addEventListener('blur', function () {
      window.setTimeout(function () {
        if (!posted && el.isConnected && stateOf(el) !== renderedOf(el)) post(el);
      }, 0);
    }, { once: true });
  }

  // Puts the snapshot's focus, caret and typing back on its replacement, and
  // returns the replacement if its value must be posted -- once htmx has
  // wired up the new form, which it does when the swap settles.
  function restoreFocus(snap) {
    var el = fieldsLike(snap.key)[snap.index];
    if (!el || !sameRow(snap, el)) return null;
    if (document.activeElement !== el) {
      try { el.focus({ preventScroll: true }); } catch (e) { el.focus(); }
    }
    var unsentTyping = snap.state !== snap.rendered && !wasPosted(snap);
    if (unsentTyping && stateOf(el) !== snap.state && canHold(el, snap.state)) {
      setState(el, snap.state);
      if (isTyped(el)) {
        select(el, snap.start, snap.end);
        postOnLeave(el);
        return null;
      }
      // A choice is made in one go, so it is posted now, as its own change
      // would have been had htmx not dropped it with the old form.
      return el;
    }
    if (stateOf(el) === snap.state) select(el, snap.start, snap.end);
    return null;
  }

  // Puts a pending change back on its replacement; returns the replacement,
  // whose value is to be posted, as restoreFocus does.
  function restoreChange(snap) {
    if (wasPosted(snap)) return null;
    var el = fieldsLike(snap.key)[snap.index];
    if (!el || !sameRow(snap, el)) return null;
    if (stateOf(el) !== snap.state) {
      // Not an option any more: the server's rendering stands, and nothing is
      // posted again.
      if (!canHold(el, snap.state)) return null;
      setState(el, snap.state);
    }
    return el;
  }

  // Fields changed since their form last posted; see `htmx:configRequest`.
  var unsent = [];

  // The hidden field every inspector form names its draft revision in, and
  // what this page's own flushes moved each revision it has seen on to.
  var REVISION = 'draft_revision';
  var revisionNow = {};

  document.addEventListener('change', function (event) {
    var el = event.target;
    if (isPosted(el) && postOf(el) && unsent.indexOf(el) < 0) unsent.push(el);
  }, true);

  document.body.addEventListener('htmx:configRequest', function (event) {
    var detail = event.detail || {};
    if (detail.headers) detail.headers['X-Editor-Client'] = CLIENT_ID;
    // A form rendered before this page saved a pending change on its own
    // (see "Saving a pending change before navigating") names the revision
    // that save made instead.
    var data = detail.formData;
    if (data && data.has && data.has(REVISION) && data.get) {
      var given = String(data.get(REVISION));
      if (revisionNow[given]) data.set(REVISION, revisionNow[given]);
    }
    // What the request posts is on its way, as it stands now.
    var from = detail.elt;
    unsent = unsent.filter(function (el) {
      return el.isConnected && !(from && (from === el || from.contains(el)));
    });
  });

  var carried = null;   // what the swap in progress must carry across

  document.body.addEventListener('htmx:beforeSwap', function (event) {
    var detail = event.detail || {};
    // A flush (below) swaps nothing, and leaves a carry in progress alone.
    if (detail.swapOverride === 'none') return;
    carried = null;
    if (!detail.target || detail.target.id !== 'editor-body' || !detail.shouldSwap) return;
    var config = detail.requestConfig || {};
    var active = document.activeElement;
    var focus = isField(active) && postOf(active) ? snapshot(active, config) : null;
    carried = {
      request: config,
      toPost: [],
      focus: focus,
      changes: unsent.filter(function (el) {
        return el.isConnected && el !== active;
      }).map(function (el) { return snapshot(el, config); }),
    };
  });

  // Put back whatever of `carried` has left the page; keep the rest (a field
  // in an open picker, which hangs off <body> until `reopenPicker` replaces
  // it) for later.
  function carryAcross(load) {
    if (!carried) return;
    if (load) {
      // Still on the page, so still editable while the swap settled.
      if (carried.focus && carried.focus.el === document.activeElement) capture(carried.focus);
      carried.changes.forEach(capture);
    }
    var focus = carried.focus;
    var toPost = carried.toPost;
    if (focus && !focus.el.isConnected) {
      carried.focus = null;
      toPost.push(restoreFocus(focus));
    }
    var gone = carried.changes.filter(function (snap) { return !snap.el.isConnected; });
    carried.changes = carried.changes.filter(function (snap) { return snap.el.isConnected; });
    gone.forEach(function (snap) { toPost.push(restoreChange(snap)); });
    unsent = unsent.filter(function (el) { return el.isConnected; });
  }

  // Straight after the swap, before a key can reach a detached field.
  document.body.addEventListener('htmx:afterSwap', function (event) {
    var detail = event.detail || {};
    if (!carried || carried.request !== detail.requestConfig) return;
    if (event.target !== detail.target) return;  // an out-of-band swap
    carryAcross(false);
  });

  /* ---------------------------------------------------------------------
   * Saving a pending change before navigating (#2348)
   *
   * A change is posted 200 ms after it is made. A click in that window that
   * replaces the inspector -- another card, "+ Actor", a place on the map --
   * took the form away before its post was sent, and only a body swap carried
   * the change across (above): an inspector swap dropped it.
   *
   * So a request that replaces `#inspector` or `#editor-body` first posts
   * every other form holding a change not posted yet, one at a time, and is
   * sent once they have been answered: the server takes them in that order.
   * Those posts are what the forms would have sent, answered with
   * `swap: 'none'` -- the page shows the navigation's answer, and behind an
   * inspector swap the canvas shows the document as it was until the next
   * edit re-renders it. The form's own debounced post is then dropped, so
   * nothing is posted twice.
   *
   * Each flush moves the draft on a revision while the page's forms still
   * name the one before. The lenient check passes them -- the saves since are
   * this page's -- but a row button is checked strictly, and would be refused
   * by the very flush its click caused. So when the answer is the revision
   * after the one the flush sent, which means no other save came between,
   * and every row the page shows is where the answer puts it, requests name
   * that revision in place of the one their form shows. A flush can move
   * rows -- a waypoint condition put on another vertex re-sorts them all --
   * and a row button rendered before it would then act on whatever row came
   * to its index; it is refused instead, as the strict check would.
   *
   * A flush that fails stops the navigation, its failure shown as any other's,
   * and the field keeps what was typed, still a change not posted: the next
   * navigation tries it again. So does one htmx would not send because the
   * browser found its form invalid (`htmx:validation:halted`, #2632): the
   * inspector forms are `novalidate`, but one that is not is not dropped.
   * One the server refuses (a value it cannot take) is posted again as an
   * ordinary edit, so its reason is shown where an edit's always is.
   * ------------------------------------------------------------------- */

  var inFlight = [];     // elements whose request is on its way
  var flushing = null;   // the flushes a navigation waits for

  document.body.addEventListener('htmx:beforeSend', function (event) {
    var elt = event.detail && event.detail.elt;
    if (elt) inFlight.push(elt);
  });

  document.body.addEventListener('htmx:afterRequest', function (event) {
    var at = inFlight.indexOf(event.detail && event.detail.elt);
    if (at >= 0) inFlight.splice(at, 1);
  });

  function navigates(detail) {
    var target = detail.target;
    return !!target && (target.id === 'inspector' || target.id === 'editor-body');
  }

  function hasUnsent(owner) {
    return unsent.some(function (el) { return el.isConnected && ownerOf(el) === owner; });
  }

  // The posting elements holding a change not posted yet, but not *elt*,
  // whose request posts its own fields, nor one whose post is on its way:
  // the swap its answer makes carries the change across.
  function pendingOwners(elt) {
    inFlight = inFlight.filter(function (el) { return el.isConnected; });
    var owners = [];
    unsent.forEach(function (el) {
      var owner = el.isConnected ? ownerOf(el) : null;
      if (!owner || owner === elt || owners.indexOf(owner) >= 0) return;
      if (inFlight.indexOf(owner) < 0) owners.push(owner);
    });
    return owners;
  }

  function parsed(text) {
    return window.DOMParser
      ? new window.DOMParser().parseFromString(String(text || ''), 'text/html')
      : null;
  }

  // The `data-row` inputs of the row-addressed forms under *root*, by the
  // URL of their form and their name.
  function rowMap(root) {
    var map = {};
    Array.prototype.forEach.call(root.querySelectorAll('form[hx-post]'), function (form) {
      var url = form.getAttribute('hx-post');
      rowInputs(form).forEach(function (input) {
        (map[url] = map[url] || {})[input.name] = input;
      });
    });
    return map;
  }

  // Whether every row this page shows sits, in the answer *page* to the
  // flush *config*, at the URL it sits at here, under its name or the one
  // the flush renamed it to: then no index a row button posts has moved.
  function rowsStay(page, config) {
    var before = rowMap(document);
    var after = rowMap(page);
    var urls = Object.keys(before);
    if (urls.length !== Object.keys(after).length) return false;
    return urls.every(function (url) {
      var now = after[url];
      return !!now && Object.keys(before[url]).every(function (name) {
        return !!now[name] && rowNames(before[url][name], config).indexOf(now[name].value) >= 0;
      });
    });
  }

  // Remember the revision a flush made, when it made the only save since the
  // one it sent and moved no row: see `htmx:configRequest`.
  function noteRevision(answer, page) {
    var config = answer.requestConfig;
    var data = config && config.formData;
    var made = page && page.querySelector('input[name="' + REVISION + '"]');
    if (!made || !data || !data.get || !data.has || !data.has(REVISION)) return;
    var sent = String(data.get(REVISION));
    if (!/^\d+$/.test(sent) || made.value !== String(Number(sent) + 1)) return;
    if (!rowsStay(page, config)) return;
    Object.keys(revisionNow).forEach(function (shown) {
      if (revisionNow[shown] === sent) revisionNow[shown] = made.value;
    });
    revisionNow[sent] = made.value;
  }

  // Keep the changes *held* as changes not posted yet, and stop the
  // navigation: the next one posts them again.
  function keep(held) {
    held.forEach(function (el) {
      if (el.isConnected && unsent.indexOf(el) < 0) unsent.push(el);
    });
    return false;
  }

  // Whether the navigation may go on after *owner*'s flush, which posted
  // the changes *held*, was *answer*ed -- or *halted* unsent by the
  // browser's constraint validation, which is no sign its form had left.
  function flushed(owner, answer, held, halted) {
    if (halted) {                           // #2632: a failure, not a send
      if (owner.reportValidity) owner.reportValidity();
      return keep(held);
    }
    if (!answer) return true;               // not sent: its form had left
    if (!answer.successful) return keep(held);  // shown by showRequestError
    var page = parsed(answer.xhr && answer.xhr.responseText);
    if (page && page.querySelector('[data-edit-refused]')) {
      window.htmx.ajax('POST', owner.getAttribute('hx-post'), { source: owner });
      return false;
    }
    noteRevision(answer, page);
    return true;
  }

  function flush(owner) {
    return new Promise(function (resolve) {
      var answer = null;
      var halted = false;
      var held = unsent.filter(function (el) { return ownerOf(el) === owner; });
      function heard(event) {
        // Its own debounced post targets `#editor-body`; the flush, itself.
        if (event.detail && event.detail.target === owner) answer = event.detail;
      }
      // htmx sends no form the browser finds invalid, and answers nothing.
      function stopped() { halted = true; }
      function done() {
        owner.removeEventListener('htmx:afterRequest', heard);
        owner.removeEventListener('htmx:validation:halted', stopped);
        resolve(flushed(owner, answer, held, halted));
      }
      owner.addEventListener('htmx:afterRequest', heard);
      owner.addEventListener('htmx:validation:halted', stopped);
      window.htmx.ajax('POST', owner.getAttribute('hx-post'),
        { source: owner, target: owner, swap: 'none' }).then(done, done);
      // After the call, which runs the flush's own `htmx:confirm`.
      owner.__flushed = true;
    });
  }

  document.body.addEventListener('htmx:confirm', function (event) {
    var detail = event.detail || {};
    var elt = detail.elt;
    var cause = detail.triggeringEvent;
    if (elt && elt.__flushed && cause && cause.type === 'change') {
      elt.__flushed = false;
      // The debounced post of a change its flush has saved.
      if (!hasUnsent(elt)) {
        event.preventDefault();
        return;
      }
    }
    if (!window.htmx || !navigates(detail) || !detail.issueRequest) return;
    var owners = pendingOwners(elt);
    if (!owners.length && !flushing) return;
    event.preventDefault();
    var run = (flushing || Promise.resolve()).then(function () {
      return owners.reduce(function (chain, owner) {
        return chain.then(function (go) {
          return go && owner.isConnected ? flush(owner) : go;
        });
      }, Promise.resolve(true));
    });
    flushing = run;
    run.then(function (go) {
      if (flushing === run) flushing = null;
      // `false`: not confirmed, so an `hx-confirm` question is still asked.
      if (go) detail.issueRequest(false);
    });
  });

  /* ---------------------------------------------------------------------
   * Failed requests
   *
   * htmx 2 swaps no 4xx/5xx response, so the error pages the server renders
   * (a draft deleted in another tab, a drafts directory that refuses writes)
   * would otherwise never be seen: the click does nothing and the field keeps
   * the typed value as if it were stored. The reason is read out of the
   * response -- never put into the page as HTML -- and shown as a notice.
   * ------------------------------------------------------------------- */

  var REQUEST_ERROR_ATTR = 'data-request-error';

  /* {title, message} of a failed request, from what the server answered:
   * the error page's `data-error-*` elements, a text/plain reason, or a JSON
   * `detail`; a generic sentence for anything else. */
  function requestError(xhr) {
    if (!xhr || !xhr.status) {
      return { title: 'Request failed', message: 'The server could not be reached.' };
    }
    var generic = {
      title: 'Request failed',
      message: 'The server answered HTTP ' + xhr.status + '.',
    };
    var type = (xhr.getResponseHeader('Content-Type') || '').toLowerCase();
    var text = String(xhr.responseText || '');
    if (type.indexOf('text/plain') === 0) {
      return text.trim() ? { title: generic.title, message: text.trim() } : generic;
    }
    if (type.indexOf('application/json') === 0) {
      try {
        var detail = JSON.parse(text).detail;
        if (typeof detail === 'string' && detail) {
          return { title: generic.title, message: detail };
        }
      } catch (e) { /* not JSON after all */ }
      return generic;
    }
    if (type.indexOf('text/html') === 0 && window.DOMParser) {
      // An inert document: nothing in it runs, and only its text is read.
      var doc = new window.DOMParser().parseFromString(text, 'text/html');
      var heading = doc.querySelector('[data-error-heading]');
      var message = doc.querySelector('[data-error-message]');
      var said = message && message.textContent.trim();
      if (said) {
        return {
          title: (heading && heading.textContent.trim()) || generic.title,
          message: said,
        };
      }
    }
    return generic;
  }
  window.editorRequestError = requestError;

  /* The verb of the request an htmx event is about, lower case ('get' when
   * htmx did not say). */
  function requestVerb(detail) {
    var config = detail && detail.requestConfig;
    return String((config && config.verb) || 'get').toLowerCase();
  }

  function errorNotice(title, message, verb, reload) {
    var notice = document.createElement('div');
    notice.className = 'ed-notice ed-notice-bad';
    notice.setAttribute('role', 'alert');
    // Which kind of request failed: see clearRequestError.
    notice.setAttribute(REQUEST_ERROR_ATTR, verb || 'get');
    notice.style.marginBottom = '12px';
    var body = document.createElement('div');
    var head = document.createElement('div');
    head.className = 'ed-notice-title';
    head.textContent = title;
    var text = document.createElement('span');
    text.textContent = message;
    body.appendChild(head);
    body.appendChild(text);
    if (reload) {
      // 409: the draft was changed in another tab, and this page's forms
      // describe the copy before that. Reloading is the way back in step.
      var again = document.createElement('button');
      again.type = 'button';
      again.className = 'ed-btn-sm';
      again.style.marginTop = '8px';
      again.textContent = 'Reload';
      again.addEventListener('click', function () { window.location.reload(); });
      body.appendChild(document.createElement('br'));
      body.appendChild(again);
    }
    notice.appendChild(body);
    return notice;
  }

  function showRequestError(event) {
    var detail = event.detail || {};
    var elt = detail.elt;
    // An element that shows its own failures (the map library's button).
    if (elt && elt.closest && elt.closest('[data-handles-errors]')) return;
    var error = requestError(detail.xhr);
    var target = detail.target;
    if (target && target.id === 'export-result') {
      // As the server's own "Export failed" answer looks; htmx has already
      // taken the spinner down.
      target.replaceChildren(errorNotice('Export failed — nothing to download',
        error.title + ': ' + error.message, requestVerb(detail)));
      return;
    }
    var stale = !!(detail.xhr && detail.xhr.status === 409);
    var notice = errorNotice(error.title, error.message, requestVerb(detail), stale);
    var status = document.getElementById('status');
    if (status) {
      status.replaceChildren(notice);
    } else {
      var shell = document.querySelector('main') || document.body;
      var old = shell.querySelector(':scope > [' + REQUEST_ERROR_ATTR + ']');
      if (old) old.remove();
      shell.insertBefore(notice, shell.firstChild);
    }
    if (notice.scrollIntoView) notice.scrollIntoView({ block: 'nearest' });
  }

  /* A request that went through takes the last failure's notice down: what it
   * said ("not saved") is no longer the state of the draft. Only a write
   * proves that of a failed write, though: with a drafts directory that
   * refuses writes, reads (opening a node in the inspector) still succeed,
   * and must not take down the notice that edits are not being stored. */
  function clearRequestError(event) {
    if (!event.detail || !event.detail.successful) return;
    var wrote = requestVerb(event.detail) !== 'get';
    var notices = document.querySelectorAll('[' + REQUEST_ERROR_ATTR + ']');
    for (var i = 0; i < notices.length; i++) {
      if (notices[i].closest('#export-result')) continue;
      if (wrote || notices[i].getAttribute(REQUEST_ERROR_ATTR) === 'get') {
        notices[i].remove();
      }
    }
  }

  /* htmx fires `htmx:responseError` on the element that sent the request, and
   * an event on an element no longer in the page reaches no listener here.
   * That is the second of two quick clicks: the answer to the first replaced
   * the body, and with it the button, before the second was refused (409, a
   * row button checked strictly). Its refusal is shown all the same. */
  document.body.addEventListener('htmx:beforeSend', function (event) {
    var detail = event.detail || {};
    var xhr = detail.xhr;
    var elt = detail.elt;
    if (!xhr || !elt) return;
    xhr.addEventListener('loadend', function () {
      if (xhr.status >= 400 && !elt.isConnected) {
        showRequestError({ detail: Object.assign({}, detail, { xhr: xhr }) });
      }
    });
  });

  document.body.addEventListener('htmx:responseError', showRequestError);
  document.body.addEventListener('htmx:sendError', showRequestError);
  document.body.addEventListener('htmx:afterRequest', clearRequestError);

  /* --------------------------------------------------------------------- */

  function refresh() {
    // Before mounting: a swap has just detached whatever was there, and the
    // replacements are about to allocate their own.
    reapViewers();
    reopenPicker();
    scheduleRedraw();
    mountMaps();
    syncPickerHighlight();
    // The pins are re-rendered by the server on every swap while the map they
    // sit over is carried across it, so they are placed again from what the
    // frame already measured.
    layoutEveryPin();
  }

  /* The overview is a panel of its own, outside `#editor-body`, because it
   * holds state a body swap would reset -- which search pattern is bound. This
   * is how it hears that the document moved: it refreshes itself on the event,
   * from a URL that carries the pattern it is showing. */
  function afterSettle(event) {
    var target = event.detail && event.detail.target;
    // htmx settles every element a response swapped in, the out-of-band ones
    // (the top bar's crumb) too, each with this same detail: answer once, for
    // the target.
    if (target && event.target !== target && target.isConnected) return;
    // A field in an open picker is replaced by `reopenPicker`, in refresh().
    var config = event.detail && event.detail.requestConfig;
    var pending = carried && carried.request === config ? carried : null;
    if (target && target.id === 'scenario-map') {
      // The panel replaced itself: its map is parked and put back, and its
      // pins are new. Nothing outside it moved, so nothing outside it is redone.
      mountMaps();
      layoutEveryPin();
      return;
    }
    if (pending) carryAcross(true);   // read what was typed before it goes
    refresh();
    if (pending) {
      carryAcross(false);
      carried = null;
      // htmx has wired up the new forms by now: post what the swap held back.
      pending.toPost.forEach(function (el) { if (el && el.isConnected) post(el); });
    }
    if (target && target.id === 'editor-body') {
      document.body.dispatchEvent(new CustomEvent('scenario-changed'));
    }
  }

  // `afterSettle` only: it fires after `afterSwap` for the same swap, and after
  // the swapped-in nodes have been laid out, which is what drawLinks measures.
  // Binding both ran every reap and mount twice per edit.
  document.body.addEventListener('htmx:afterSettle', afterSettle);
  window.addEventListener('resize', function () {
    scheduleRedraw();
    schedulePinLayout();
  });
  refresh();
})();
