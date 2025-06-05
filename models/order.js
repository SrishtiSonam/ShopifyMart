const mongoose = require('mongoose');

const orderSchema = new mongoose.Schema({
  id: Number,
  email: String,
  total_price: String,
  created_at: String,
  currency: String,
  customer: Object,
  line_items: Array
});

module.exports = mongoose.model('Order', orderSchema);
