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
   * Each trigger block is linked to its action card. The line is drawn into
   * one SVG overlay positioned over the canvas, so adding a condition never
   * has to reason about layout — it just re-renders and we redraw.
   * ------------------------------------------------------------------- */

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
    scroll.querySelectorAll('.ed-trigger-wrap[data-links-to]').forEach(function (wrap) {
      var target = document.getElementById(wrap.getAttribute('data-links-to'));
      if (!target) return;
      var trigger = wrap.querySelector('.ed-trigger');
      if (!trigger) return;

      var from = trigger.getBoundingClientRect();
      var to = target.getBoundingClientRect();
      var x1 = from.left + from.width / 2 - origin.left + scroll.scrollLeft;
      var y1 = from.top - origin.top + scroll.scrollTop;
      var x2 = to.left + to.width / 2 - origin.left + scroll.scrollLeft;
      var y2 = to.bottom - origin.top + scroll.scrollTop;
      if (y1 <= y2) return; // trigger is not below its action; nothing to draw

      var mid = (y1 + y2) / 2;
      var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', 'M ' + x1 + ' ' + y1 + ' C ' + x1 + ' ' + mid + ', ' +
        x2 + ' ' + mid + ', ' + x2 + ' ' + (y2 + 5));
      svg.appendChild(path);

      var head = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      head.setAttribute('d', 'M ' + x2 + ' ' + y2 + ' l -4 5.5 l 8 0 z');
      head.setAttribute('class', 'head');
      svg.appendChild(head);
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
   * The module is optional. If it cannot be loaded — no network, a locked-down
   * machine — the server-rendered SVG that ships alongside it stays visible and
   * the match count is unaffected.
   * ------------------------------------------------------------------- */

  var viewerModule = null;   // resolved module, or false once loading has failed

  function loadViewerModule(url) {
    if (viewerModule === false) return Promise.resolve(null);
    if (viewerModule) return viewerModule;
    viewerModule = import(/* webpackIgnore: true */ url).catch(function (error) {
      console.warn('Lanelet2 map viewer unavailable, keeping the SVG preview:', error);
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

    loadViewerModule(url).then(function (module) {
      if (!module || !module.LaneletViewer) return;

      // Revealed only now: the viewer needs a sized box to mount into, and an
      // empty one is what an unreachable module would otherwise leave behind.
      frame.hidden = false;
      var hint = document.querySelector('[data-viewer-hint]');
      if (hint) hint.hidden = false;

      var viewer = new module.LaneletViewer(frame, {
        theme: 'light',
        background: 'transparent',
        scalebar: true,
      });
      frame.__viewer = viewer;

      viewer.addEventListener('load', function () {
        // The fallback drawing has served its purpose the moment the real map
        // is on screen; leaving both would just be two maps.
        var fallback = document.querySelector('[data-map-fallback]');
        if (fallback) fallback.hidden = true;

        // The fitted overview is the useful view here: highlighting is what
        // shows where the matches are, and zooming to the current spawn would
        // throw away the very thing the preview is for.
        var matched = (frame.dataset.matched || '')
          .split(',')
          .map(function (v) { return parseInt(v, 10); })
          .filter(function (v) { return !isNaN(v); });
        var selected = parseInt(frame.dataset.selected, 10);
        if (!isNaN(selected)) matched.push(selected);
        if (matched.length) viewer.setHighlight(matched);
      });

      // Picking a lanelet on the map is the fastest way to set a spawn, and it
      // goes through the same endpoint the number field does.
      viewer.addEventListener('select', function (event) {
        var picked = event.detail && event.detail.id;
        if (!picked || !window.htmx) return;
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
    document.querySelectorAll('.ed-map-frame[data-map-viewer]').forEach(mountMap);
  }

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
