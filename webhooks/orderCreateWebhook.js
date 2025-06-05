const crypto = require('crypto');
const amqp = require('amqplib');

async function pushToQueue(orderData) {
  const conn = await amqp.connect('amqp://localhost');
  const channel = await conn.createChannel();
  const queue = 'order_create_queue';

  await channel.assertQueue(queue, { durable: true });
  channel.sendToQueue(queue, Buffer.from(JSON.stringify(orderData)), {
    persistent: true
  });

  console.log("Order pushed to queue:", orderData.id);
}

module.exports = async (req, res) => {
  try {
    const hmacHeader = req.get('X-Shopify-Hmac-Sha256');
    const rawBody = req.body;

    const generatedHmac = crypto
      .createHmac('sha256', process.env.WEBHOOK_SECRET)
      .update(rawBody)
      .digest('base64');

    if (generatedHmac !== hmacHeader) {
      console.warn('Webhook Verification Failed: Invalid HMAC');
      return res.status(401).send('Unauthorized');
    }

    const parsedOrder = JSON.parse(rawBody.toString('utf8'));
    console.log('Verified Order Create Webhook:', parsedOrder.id);

    await pushToQueue(parsedOrder);

    return res.status(200).send('Order received and queued');
  } catch (error) {
    console.error('Error handling order create webhook:', error.message);
    return res.status(400).send('Bad Request');
  }
};
