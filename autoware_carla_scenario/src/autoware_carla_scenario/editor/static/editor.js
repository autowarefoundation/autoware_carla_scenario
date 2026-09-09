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

  /* The notices that belong to one map: its "could not be loaded", its caption.
     Scoped by `data-viewer-scope`, which every template that mounts a viewer
     marks its container with -- there is more than one map on the page now, and
     a document-wide lookup would reveal the picker's notice for the overview's
     failure. Resolved when it is needed rather than captured at mount: under
     reuse the frame outlives several renders of the fragment around it. */
  function previewOf(frame) {
    return frame.closest('[data-viewer-scope]') || document;
  }

  function revealHint(frame) {
    var hint = previewOf(frame).querySelector('[data-viewer-hint]');
    if (hint) hint.hidden = false;
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
   * for the same map. A picker has no key: it is mounted when someone opens it
   * and destroyed when they close it, which is already once per deliberate act.
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
    // The places the overview draws move when another search pattern is bound,
    // so what a click on one of them opens moves with them.
    if (fresh.dataset.owners) live.dataset.owners = fresh.dataset.owners;
    if (live.__loaded) {
      applyHighlight(live);
      revealHint(live);
    }
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
        revealHint(frame);

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
        if (!probing) layoutPins(frame);
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

        // The overview above the timeline is not a picker: it draws places the
        // document already names, so the one thing a click there can mean is
        // "show me what named this". A place is still edited where it is
        // written -- in that object's own inspector -- so nothing is saved here.
        if (frame.dataset.opensInspector) {
          openOwner(frame, picked);
          return;
        }

        // Otherwise the map is a picker's: the field it writes into is what the
        // map was opened from.
        var input = document.getElementById(frame.dataset.picksInto || '');
        if (!input) return;

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
    var scope = frame.closest('[data-viewer-scope]');
    return scope ? scope.querySelector('[data-pins]') : null;
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

    // The frame's border box, because that is the box the viewer sizes itself
    // from and reports `getView` against.
    var box = frame.getBoundingClientRect();
    var view = viewer.getView();
    var width = box.width;
    var height = box.height;
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
      pin.hidden = false;
      // A label reads away from its dot, and on the right of the map that runs
      // it off the edge -- or under the viewer's own zoom buttons. Past the
      // two-thirds line it reads back towards the middle instead.
      // Labels on the right half read back towards the middle -- the map's
      // right edge is also where the viewer keeps its own zoom buttons -- and
      // one that would overflow anyway is flipped wherever it sits.
      pin.classList.remove('is-flipped');
      var flipped = x > width * 0.5 || x + pin.offsetWidth > width - 6;
      pin.classList.toggle('is-flipped', flipped);
      pin.style.transform = 'translate(' + Math.round(x) + 'px, ' + Math.round(y) +
        'px)' + (flipped ? ' translateX(-100%)' : '');
    });
    layer.hidden = false;
  }

  /* Move the view to hold every place this scenario names, once. */
  function frameOnPins(frame) {
    var viewer = frame.__viewer;
    var layer = pinLayer(frame);
    if (!viewer || !layer) return;

    var xs = [];
    var ys = [];
    layer.querySelectorAll('[data-lanelet]').forEach(function (pin) {
      var at = frame.__pinAt && frame.__pinAt[pin.dataset.lanelet];
      if (at) {
        xs.push(at.x);
        ys.push(at.y);
      }
    });
    if (!xs.length) return;

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

  /* Open the inspector on whatever named a lanelet the overview drew. */
  function openOwner(frame, lanelet) {
    var owners;
    try {
      owners = JSON.parse(frame.dataset.owners || '{}');
    } catch (error) {
      return;
    }
    var owner = owners[String(lanelet)];
    if (!owner || !window.htmx) return;
    window.htmx.ajax('GET', frame.dataset.opensInspector + owner, {
      target: '#inspector',
      swap: 'innerHTML',
    });
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
      var overview = document.querySelector('.ed-map-frame[data-owners]');
      if (overview && overview.__viewer && wanted > 0) {
        overview.__viewer.focusOn(wanted, { fraction: 0.3 });
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
   * ------------------------------------------------------------------- */

  function selectable() {
    return Array.prototype.slice.call(
      document.querySelectorAll('#canvas [data-object-id]')
    );
  }

  function focusOffset(delta) {
    var nodes = selectable();
    if (!nodes.length) return;
    var index = nodes.indexOf(document.activeElement);
    var next = nodes[Math.max(0, Math.min(nodes.length - 1, index < 0 ? 0 : index + delta))];
    if (next) {
      next.focus();
      if (window.htmx) window.htmx.trigger(next, 'click');
    }
  }

  document.addEventListener('keydown', function (event) {
    if (event.target && /^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName)) return;

    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      focusOffset(1);
      event.preventDefault();
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      focusOffset(-1);
      event.preventDefault();
    } else if (event.key === 'Enter' || event.key === ' ') {
      var active = document.activeElement;
      if (active && active.hasAttribute('data-object-id')) {
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
    refresh();
    var target = event.detail && event.detail.target;
    if (target && target.id === 'editor-body') {
      document.body.dispatchEvent(new CustomEvent('scenario-changed'));
    }
  }

  // `afterSettle` only: it fires after `afterSwap` for the same swap, and after
  // the swapped-in nodes have been laid out, which is what drawLinks measures.
  // Binding both ran every reap and mount twice per edit.
  document.body.addEventListener('htmx:afterSettle', afterSettle);
  window.addEventListener('resize', scheduleRedraw);
  window.addEventListener('resize', layoutEveryPin);
  refresh();
})();
