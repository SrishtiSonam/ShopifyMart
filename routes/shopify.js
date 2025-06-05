const express = require('express');
const axios = require('axios');
const router = express.Router();

const customerQueries = require('../graphql/customerQueries');
const orderQueries = require('../graphql/orderQueries');
const validateOrder = require('../middlewares/validateOrder');

const endpoint = `https://${process.env.SHOPIFY_STORE_DOMAIN}/admin/api/${process.env.SHOPIFY_API_VERSION}/graphql.json`;

const shopifyHeaders = {
  'Content-Type': 'application/json',
  'X-Shopify-Access-Token': process.env.SHOPIFY_ADMIN_ACCESS_TOKEN,
};


router.post('/customer', async (req, res) => {
  try {
    const response = await axios.post(endpoint, {
      query: customerQueries.createCustomer,
      variables: { input: req.body }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    console.error("Error:", err.response?.data || err.message);
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

router.get('/customer/:id', async (req, res) => {
  const id = `gid://shopify/Customer/${req.params.id}`;
  try {
    const response = await axios.post(endpoint, {
      query: customerQueries.getCustomer,
      variables: { id }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

router.put('/customer/:id', async (req, res) => {
  const input = {
    id: `gid://shopify/Customer/${req.params.id}`,
    ...req.body
  };

  try {
    const response = await axios.post(endpoint, {
      query: customerQueries.updateCustomer,
      variables: { input }
    }, { headers: shopifyHeaders });

    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

router.delete('/customer/:id', async (req, res) => {
  const input = { id: `gid://shopify/Customer/${req.params.id}` };
  try {
    const response = await axios.post(endpoint, {
      query: customerQueries.deleteCustomer,
      variables: { input }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

//CREATE ORDER
router.post('/order', validateOrder, async (req, res) => {
  const input = req.body.input;

  if (!input || !input.lineItems || !Array.isArray(input.lineItems) || input.lineItems.length === 0) {
    return res.status(400).json({ error: "Missing or invalid lineItems in input" });
  }

  try {
    // Step 1: Create draft order
    const response = await axios.post(endpoint, {
      query: orderQueries.draftOrderCreate,
      variables: { input }
    }, { headers: shopifyHeaders });

    const draftOrder = response.data.data?.draftOrderCreate?.draftOrder;
    const userErrors = response.data.data?.draftOrderCreate?.userErrors;

    if (!draftOrder) {
      return res.status(400).json({
        error: "Failed to create draft order",
        userErrors
      });
    }

    // Step 2: Complete draft order
    const completeResponse = await axios.post(endpoint, {
      query: orderQueries.draftOrderComplete,
      variables: { id: draftOrder.id }
    }, { headers: shopifyHeaders });

    const completedOrder = completeResponse.data.data?.draftOrderComplete?.draftOrder;
    const completionErrors = completeResponse.data.data?.draftOrderComplete?.userErrors;

    if (!completedOrder) {
      return res.status(400).json({
        error: "Failed to complete draft order",
        completionErrors
      });
    }

    res.json({
      draftOrder,
      completedOrder
    });

  } catch (err) {
    console.error("Shopify API Error:", err.response?.data || err.message);
    res.status(500).json({ error: err.response?.data || err.message });
  }
});



// FETCH ORDER
router.get('/order/:id', async (req, res) => {
  const id = `gid://shopify/Order/${req.params.id}`;
  try {
    const response = await axios.post(endpoint, {
      query: orderQueries.getOrder,
      variables: { id }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

// Update 
router.put('/order/:id', async (req, res) => {
  const input = {
    id: `gid://shopify/Order/${req.params.id}`,
    ...req.body 
  };
  try {
    const response = await axios.post(endpoint, {
      query: orderQueries.updateOrder,
      variables: { input }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

// Close Order
router.post('/order/:id/close', async (req, res) => {
  const id = `gid://shopify/Order/${req.params.id}`;
  try {
    const response = await axios.post(endpoint, {
      query: orderQueries.closeOrder,
      variables: { id }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.response?.data || err.message });
  }
});

// Cancel Order
router.post('/order/:id/cancel', async (req, res) => {
  const id = `gid://shopify/Order/${req.params.id}`;
  const reason = req.body.reason || "CUSTOMER"; 
  try {
    const response = await axios.post(endpoint, {
      query: orderQueries.cancelOrder,
      variables: { id, reason }
    }, { headers: shopifyHeaders });
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: err.response?.data || err.message });
  }
});



module.exports = router;
