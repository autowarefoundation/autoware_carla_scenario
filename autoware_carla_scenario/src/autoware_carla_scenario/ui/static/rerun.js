// Shows cases' Rerun recordings in Rerun's web viewer (@rerun-io/web-viewer).
//
// mountRerun(element, {viewerUrl, makeUrls, status}) asks the viewer app for
// each case's recording (POST makeUrls[i]: it converts the trajectory log if
// there is no recording yet), then starts one web viewer on `element` with
// all of them: one recording, or several to compare (the viewer's recording
// panel switches between them). `viewerUrl` is the package's directory
// (index.js in it), served by the app or a CDN, of the Rerun version the
// recordings are written with.

function say(status, text, kind) {
  if (!status) return;
  status.textContent = text;
  status.dataset.kind = kind || 'info';
  status.hidden = !text;
}

async function makeRecording(url) {
  const response = await fetch(url, { method: 'POST' });
  let body = {};
  try { body = await response.json(); } catch (e) { /* not JSON */ }
  if (!response.ok) {
    const error = new Error(body.error || `${response.status} ${response.statusText}`);
    error.missingConverter = Boolean(body.missing_converter);
    error.installHint = body.install_hint || '';
    throw error;
  }
  return body;
}

export async function mountRerun(element, { viewerUrl, makeUrls, status, onError }) {
  say(status, makeUrls.length > 1
    ? `Preparing ${makeUrls.length} recordings...` : 'Preparing the recording...');
  let made;
  try {
    made = await Promise.all(makeUrls.map(makeRecording));
  } catch (error) {
    say(status, error.message, 'error');
    if (onError) onError(error);
    return null;
  }
  if (!viewerUrl) {
    say(status, 'No web viewer is embedded (VIEWER_RERUN_WEB_VIEWER=none).', 'info');
    return null;
  }
  say(status, 'Loading the Rerun web viewer...');
  try {
    const { WebViewer } = await import(`${viewerUrl}/index.js`);
    const viewer = new WebViewer();
    const urls = made.map(m => new URL(m.url, window.location.href).href);
    await viewer.start(urls.length === 1 ? urls[0] : urls, element, {
      hide_welcome_screen: true,
      width: '100%',
      height: '100%',
    });
    element.dataset.rerunStarted = String(urls.length);
    say(status, '');
    return viewer;
  } catch (error) {
    say(status, `The Rerun web viewer could not start: ${error.message || error}`, 'error');
    if (onError) onError(error);
    return null;
  }
}
