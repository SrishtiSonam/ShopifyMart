require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const https = require('https');
const path = require('path');

const app = express();

const shopifyRoutes = require('./routes/shopify');
const orderCreateWebhook = require('./webhooks/orderCreateWebhook');

const sslOptions = {
  key: fs.readFileSync(path.join(__dirname, 'certs', 'key.pem')),
  cert: fs.readFileSync(path.join(__dirname, 'certs', 'cert.pem')),
};

app.use('/shopify', bodyParser.json(), shopifyRoutes);

app.post(
  '/webhooks/orders/create',
  express.raw({ type: 'application/json' }),
  orderCreateWebhook
);

const PORT = process.env.PORT || 3000;
https.createServer(sslOptions, app).listen(PORT, () => {
  console.log(`HTTPS server running at https://localhost:${PORT}`);
});
