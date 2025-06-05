const crypto = require('crypto');
const fs = require('fs');
require('dotenv').config();

const payload = fs.readFileSync('./test_payload.json');
const secret = process.env.WEBHOOK_SECRET;

const hmac = crypto
  .createHmac('sha256', secret)
  .update(payload)
  .digest('base64');

console.log('HMAC for Postman Header:', hmac);
