# ShopifyMart

It is a backend integration project for Shopify that allows merchants and developers to perform complete CRUD operations on Customers and Orders using the Shopify Admin GraphQL API. It also supports real-time webhook handling for order creation and pushes order data into a queue for background processing using RabbitMQ. MongoDB is used as the primary database for persisting customer and order data.


## 2. Features

- ✅ **Postman-tested API endpoints**
- ✅ **MongoDB for persistent storage**
- ✅ **Order CRUD via Shopify GraphQL API**
- ✅ **HTTPS support with SSL certificates**
- ✅ **Draft order creation and confirmation**
- ✅ **Customer CRUD via Shopify GraphQL API**
- ✅ **Webhook listener for order creation events**
- ✅ **RabbitMQ queue integration to handle order data in the background**

---

## 3. Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Backend     | Node.js + Express                   |
| Shopify API | Admin GraphQL API                   |
| Queue       | RabbitMQ (AMQP protocol)            |
| Database    | MongoDB                             |
| HTTPS       | Self-signed SSL (certs folder)      |
| Testing     | Postman                             |

---

## 4. Folder Structure
```
shopSyncer/
├── certs/                 # SSL certificates (not committed)
├── graphql/               # GraphQL queries for Customer and Order
│   ├── customerQueries.js
│   └── orderQueries.js
├── queue/
│   └── consumer.js        # Consumes messages from RabbitMQ
├── middlewares/
│   └── validateOrder.js
├── models/
│   └── order.js
├── routes/
│   └── shopify.js         # All CRUD endpoints
├── webhooks/
│   └── orderCreateWebhook.js  # Webhook for order creation
├── .env                   # Environment variables (not committed)
├── .gitignore
├── index.js               # Main HTTPS server entry
├── package.json
└── README.md

```

---

## 5. Webhook Flow

1. Shopify sends an `orders/create` event to `/webhooks/orders/create`.
2. The webhook is verified using HMAC SHA-256.
3. The verified payload is pushed to a RabbitMQ queue (`order_create_queue`).
4. The consumer (`queue/consumer.js`) listens to this queue and saves the order in MongoDB.

---

## 6. Security Practices

- `.env` and `certs/` are added to `.gitignore` to prevent accidental leakage.
- Webhooks are verified using HMAC and the shared `WEBHOOK_SECRET`.
- HTTPS server uses SSL for secure local testing.

---

## 7. Testing with Postman

- Use POST requests to test:
  - `/shopify/customer/create`, `/shopify/order/create`, etc.
  - Webhook endpoint: `/webhooks/orders/create` with raw `application/json` and proper HMAC header.
- You can generate HMAC using `generateHmac.js` with your payload and secret.

---

## 8. Setup Instructions

### 1. Clone the repo:
   ```bash
   git clone
```
### 2. Install dependencies

```bash
npm install 
```
### 3. Modify .env file

### 4. Start RabbitMQ locally

### 5. Run the consumer and server
```bash
node queue/consumer.js
node index.js
```
