module.exports = function validateOrder(req, res, next) {
  const input = req.body.input;
  if (!input) {
    return res.status(400).json({ error: "Missing input in request body" });
  }

  const { lineItems, customerId, shippingAddress, billingAddress } = input;

  if (!lineItems || !Array.isArray(lineItems) || lineItems.length === 0) {
    return res.status(400).json({ error: 'lineItems must be a non-empty array' });
  }

  if (!customerId) {
    return res.status(400).json({ error: 'customerId is required' });
  }

  if (!shippingAddress || !shippingAddress.address1 || !shippingAddress.city) {
    return res.status(400).json({ error: 'Valid shippingAddress is required' });
  }

  if (!billingAddress || !billingAddress.address1 || !billingAddress.city) {
    return res.status(400).json({ error: 'Valid billingAddress is required' });
  }

  next();
};
