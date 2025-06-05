const amqp = require('amqplib');
const mongoose = require('mongoose');
const Order = require('../models/Order');

async function startConsumer() {
  await mongoose.connect('mongodb://localhost:27017/shopifyApp');

  const conn = await amqp.connect('amqp://localhost');
  const channel = await conn.createChannel();
  const queue = 'order_create_queue';

  await channel.assertQueue(queue, { durable: true });

  console.log("Consumer is running and listening for new order messages...");

  channel.consume(queue, async (msg) => {
    if (msg !== null) {
      const order = JSON.parse(msg.content.toString());
      console.log(" Consuming order:", order.id);

      try {
        await Order.create(order);
        console.log("Order saved to DB");
        channel.ack(msg);
      } catch (err) {
        console.error("DB insert failed:", err.message);
      }
    }
  });
}

startConsumer().catch((err) => {
  console.error("Consumer startup failed:", err.message);
});
