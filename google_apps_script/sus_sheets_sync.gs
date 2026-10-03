/**
 * sus_sheets_sync.gs — receives CeduSync SUS survey responses from Supabase
 * (Edge Function "sus-sheets-sync") and writes them to the "Responses" tab.
 *
 * Paste into the "CeduSync SUS Survey Results" sheet (Extensions > Apps Script)
 * and deploy as a web app: Execute as "Me", Who has access "Anyone".
 *
 * The shared secret is NOT in this file. Set it once under
 * Project Settings > Script Properties: property SHARED_SECRET, value = the
 * same text you save as SHEETS_SHARED_SECRET in Supabase.
 *
 * Writes only columns A to R. Column S holds the SUS-score ARRAYFORMULA and
 * the Summary / Column guide tabs are never touched.
 */

var SCRIPT_VERSION = '2';
// "CeduSync SUS Survey Results". Opened by ID so the script works whether it
// was created from the sheet (Extensions > Apps Script) or as its own project.
var SPREADSHEET_ID = '1lKh9XoH97sdO-93M1178yL10C3Muhgw_k3sE6MtHjGQ';
var SHEET_NAME = 'Responses';
var COLUMN_COUNT = 18; // A..R
var TIME_ZONE = 'Asia/Manila';
var TEST_NAME = 'SUS TEST'; // respondent name used by Phase 5 test rows

var ROLE_LABELS = {
  barangay_secretary: 'Barangay secretary',
  cedula_personnel: 'Cedula-processing personnel',
  citizen_applicant: 'Citizen applicant'
};
var EXPERIENCE_LABELS = { none: 'None', basic: 'Basic', comfortable: 'Comfortable', advanced: 'Advanced' };
var FEATURE_LABELS = {
  voice_input: 'Voice input',
  qr_prefill: 'QR pre-fill',
  form_generation: 'Form generation',
  receipt: 'Receipt',
  voice_feedback: 'Voice feedback'
};
var PLATFORM_LABELS = { web: 'Web', mobile: 'Mobile' };

/** Open the web app URL in a browser to check the right code is deployed. */
function doGet() {
  return reply_({
    ok: true,
    service: 'CeduSync SUS sync',
    version: SCRIPT_VERSION,
    secretConfigured: Boolean(PropertiesService.getScriptProperties().getProperty('SHARED_SECRET'))
  });
}

function responsesSheet_() {
  var sheet = SpreadsheetApp.openById(SPREADSHEET_ID).getSheetByName(SHEET_NAME);
  if (!sheet) throw new Error('Tab "' + SHEET_NAME + '" not found');
  return sheet;
}

function doPost(e) {
  var body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return reply_({ ok: false, error: 'Request body is not JSON' });
  }

  var expected = PropertiesService.getScriptProperties().getProperty('SHARED_SECRET');
  if (!expected) return reply_({ ok: false, error: 'SHARED_SECRET script property is not set' });
  if (!body || body.secret !== expected) return reply_({ ok: false, error: 'Unauthorized' });

  var lock = LockService.getScriptLock();
  if (!lock.tryLock(30000)) return reply_({ ok: false, error: 'Sheet is busy, try again' });
  try {
    var response = body.response || {};
    var row = buildRow_(response);
    var sheet = responsesSheet_();

    var lastRow = lastResponseRow_(sheet);
    if (lastRow >= 2) {
      var ids = sheet.getRange(2, 2, lastRow - 1, 1).getDisplayValues();
      for (var i = 0; i < ids.length; i++) {
        if (ids[i][0] === response.id) return reply_({ ok: true, status: 'duplicate', row: i + 2 });
      }
    }

    var target = lastRow + 1;
    if (target > sheet.getMaxRows()) sheet.insertRowsAfter(sheet.getMaxRows(), 50);
    sheet.getRange(target, 1).setNumberFormat('@'); // keep the timestamp as text
    sheet.getRange(target, 1, 1, COLUMN_COUNT).setValues([row]);
    SpreadsheetApp.flush();
    return reply_({ ok: true, status: 'appended', row: target });
  } catch (err) {
    return reply_({ ok: false, error: String(err && err.message ? err.message : err) });
  } finally {
    lock.releaseLock();
  }
}

/** The 18 values for columns A..R, in the order the Column guide describes. */
function buildRow_(r) {
  if (!r.id) throw new Error('Missing response id');
  var role = ROLE_LABELS[r.role];
  var experience = EXPERIENCE_LABELS[r.computer_experience];
  var platform = PLATFORM_LABELS[r.platform];
  if (!role) throw new Error('Unknown role: ' + r.role);
  if (!experience) throw new Error('Unknown computer experience: ' + r.computer_experience);
  if (!platform) throw new Error('Unknown platform: ' + r.platform);

  var features = (r.features_used || []).map(function (code) {
    if (!FEATURE_LABELS[code]) throw new Error('Unknown feature: ' + code);
    return FEATURE_LABELS[code];
  });

  var ratings = [];
  for (var q = 1; q <= 10; q++) {
    var value = Number(r['q' + q]);
    if (!(value >= 1 && value <= 5 && Math.floor(value) === value)) throw new Error('Q' + q + ' must be 1 to 5');
    ratings.push(value);
  }

  var created = new Date(r.created_at);
  if (isNaN(created.getTime())) throw new Error('Invalid created_at');

  return [
    Utilities.formatDate(created, TIME_ZONE, 'yyyy-MM-dd HH:mm'), // A Timestamp
    String(r.id),                                                  // B Response ID
    asText_(r.respondent_name),                                    // C Name
    role,                                                          // D Role
    experience,                                                    // E Computer experience
    features.join(', '),                                           // F Features used
    platform                                                       // G Platform
  ].concat(ratings, [asText_(r.comments)]);                        // H..Q, R Comments
}

/** Free text is stored as plain text, never run as a formula. */
function asText_(value) {
  var text = value == null ? '' : String(value);
  return /^[=+\-@]/.test(text) ? "'" + text : text;
}

/** Last row with a Response ID. Column S's ARRAYFORMULA fills rows far below
 *  the data, so getLastRow()/appendRow() would land in the wrong place. */
function lastResponseRow_(sheet) {
  var maxRows = sheet.getMaxRows();
  if (sheet.getRange(maxRows, 2).getDisplayValue() !== '') return maxRows;
  return sheet.getRange(maxRows, 2).getNextDataCell(SpreadsheetApp.Direction.UP).getRow();
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

/**
 * Phase 5 cleanup: run from the Apps Script editor (select deleteTestRows, Run).
 * Clears columns A to R of rows whose Name is "SUS TEST". The header row and
 * column S are left untouched.
 */
function deleteTestRows() {
  var sheet = responsesSheet_();
  var lastRow = lastResponseRow_(sheet);
  if (lastRow < 2) return Logger.log('No response rows.');
  var names = sheet.getRange(2, 3, lastRow - 1, 1).getDisplayValues();
  var cleared = 0;
  for (var i = 0; i < names.length; i++) {
    if (names[i][0] === TEST_NAME) {
      sheet.getRange(i + 2, 1, 1, COLUMN_COUNT).clearContent();
      cleared++;
    }
  }
  Logger.log('Cleared ' + cleared + ' test row(s).');
}
