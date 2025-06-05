module.exports = {
  createCustomer: `
    mutation CreateCustomer($input: CustomerInput!) {
      customerCreate(input: $input) {
        customer {
          id
          firstName
          lastName
          email
        }
        userErrors {
          field
          message
        }
      }
    }
  `,
  getCustomer: `
    query GetCustomer($id: ID!) {
      customer(id: $id) {
        id
        firstName
        lastName
        email
      }
    }
  `,
updateCustomer: `
mutation customerUpdate($input: CustomerInput!) {
  customerUpdate(input: $input) {
    customer {
      id
      firstName
      lastName
      email
    }
    userErrors {
      field
      message
    }
  }
}
`,
 deleteCustomer: `
mutation DeleteCustomer($input: CustomerDeleteInput!) {
  customerDelete(input: $input) {
    deletedCustomerId
    userErrors {
      field
      message
    }
  }
}
`
};
