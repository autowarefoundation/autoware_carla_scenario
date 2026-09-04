/* Scenario Editor — the behaviour a server render cannot do.
 *
 * Everything that changes the scenario goes through htmx and a server render, so
 * this file is limited to three things: drawing the connector between a trigger
 * and the action it fires, moving keyboard focus around the swimlanes, and
 * mounting the Lanelet2 map viewer for the spawn preview.
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
    svg.setAttribute('width', scroll.scrollWidth);
    svg.setAttribute('height', scroll.scrollHeight);

    var origin = scroll.getBoundingClientRect();
    function toX(clientX) { return clientX - origin.left + scroll.scrollLeft; }
    function toY(clientY) { return clientY - origin.top + scroll.scrollTop; }

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
      makePath(svg, 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + mid + ', ' +
        x2 + ' ' + mid + ', ' + x2 + ' ' + (y2 + 5));
      makePath(svg, 'M ' + x2 + ' ' + y2 + ' l -4 5.5 l 8 0 z', 'head');
    });

    scroll.querySelectorAll('.ed-cond[data-caused-by]').forEach(function (card) {
      var causes = (card.getAttribute('data-caused-by') || '').split(',');
      var cardBox = card.getBoundingClientRect();

      causes.forEach(function (actionId) {
        if (!actionId) return;
        var clip = document.getElementById('node-' + actionId);
        if (!clip) return;
        var clipBox = clip.getBoundingClientRect();

        // The line wears the track colour of whoever runs the causing action.
        var lane = clip.closest('.ed-lane');
        var hue = lane
          ? getComputedStyle(lane).getPropertyValue('--lane-hue').trim()
          : '';

        var x1 = toX(clipBox.right);
        var y1 = toY(clipBox.top + clipBox.height / 2);
        var x2 = toX(cardBox.left - 6);
        var y2 = toY(cardBox.top + cardBox.height / 2);
        var bend = Math.max(28, Math.abs(x2 - x1) / 2);

        makePath(svg, 'M ' + x1 + ' ' + y1 + ' C ' + (x1 + bend) + ' ' + y1 + ', ' +
          (x2 - bend) + ' ' + y2 + ', ' + x2 + ' ' + y2, 'causes', hue);
        makePath(svg, 'M ' + (x2 + 5) + ' ' + y2 + ' l -6 -4 l 0 8 z',
          'causes-head', hue, true);
      });
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

  function mountMap(frame) {
    if (frame.dataset.mounted === '1') return;
    var url = frame.dataset.mapViewer;
    if (!url) return;
    frame.dataset.mounted = '1';

    // Captured now, while this fragment is certainly in the DOM, and scoped to
    // this preview rather than the whole page: an async callback looking these
    // up later can run against a fragment htmx has already replaced.
    var preview = frame.closest('.ed-preview, .ed-modal-body') || document;
    var key = preview.querySelector('[data-viewer-hint]');
    var unavailable = preview.querySelector('[data-viewer-unavailable]');

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

      viewer.addEventListener('load', function () {
        // Revealed together: the map appears and its caption with it, so the
        // caption never describes a drawing that is not there.
        frame.classList.remove('is-mounting');
        if (key) key.hidden = false;

        // The fitted overview is the useful view here: highlighting is what
        // shows where the matches are, and zooming to the current spawn would
        // throw away the very thing the preview is for.
        //
        // One list, because `setHighlight` is one outline colour. The server
        // decides what it means — the matches under a constraint search, the
        // pinned lanelet under a fixed spawn — so the drawing and the caption
        // below it cannot disagree.
        var highlight = ids(frame.dataset.highlight);
        if (highlight.length) viewer.setHighlight(highlight);
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

        var into = frame.dataset.picksInto;
        if (into) {
          var input = document.getElementById(into);
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
          return;
        }

        if (!window.htmx) return;
        var draft = frame.dataset.draft;
        var entity = frame.dataset.entity;
        if (!draft || !entity) return;
        window.htmx.ajax('POST', '/draft/' + draft + '/entity/' + entity, {
          target: '#editor-body',
          swap: 'innerHTML',
          values: { spawn_lanelet_id: String(picked) },
        });
      });

      viewer.loadUrl(frame.dataset.mapSrc);
    });
  }

  function mountMaps() {
    document.querySelectorAll('.ed-map-frame[data-map-viewer]').forEach(function (frame) {
      // A picker inside a closed modal is not on screen and must not pay for a
      // wasm parse of the whole map until someone asks to see it.
      if (frame.closest('.ed-modal[hidden]')) return;
      mountMap(frame);
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

  function closePickers() {
    document.querySelectorAll('[data-portalled]').forEach(function (modal) {
      modal.remove();
    });
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
      var modal = document.getElementById(opener.getAttribute('data-open-picker'));
      if (modal) {
        closePickers();
        modal.dataset.portalled = '1';
        document.body.appendChild(modal);
        modal.hidden = false;
        mountMaps();
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
    scheduleRedraw();
    mountMaps();
  }

  document.addEventListener('DOMContentLoaded', refresh);
  document.body.addEventListener('htmx:afterSwap', refresh);
  document.body.addEventListener('htmx:afterSettle', refresh);
  window.addEventListener('resize', scheduleRedraw);
  refresh();
})();
