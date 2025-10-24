module.exports = {
  extends: [
    'react-app',
    'react-app/jest'
  ],
  rules: {
    // Add any custom rules here
  },
  settings: {
    react: {
      version: 'detect'
    }
  },
  env: {
    browser: true,
    es2021: true,
    node: true,
    jest: true
  },
  parserOptions: {
    ecmaFeatures: {
      jsx: true
    },
    ecmaVersion: 12,
    sourceType: 'module'
  }
};
