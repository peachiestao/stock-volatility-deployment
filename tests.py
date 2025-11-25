import unittest
import json
from deploy import app, ALLOWED_TICKERS  # <-- CHANGED THIS LINE

class FlaskIntegrationTest(unittest.TestCase):

    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_status_code(self):
        """Test 1: Ensure the home page loads successfully."""
        print("\nRunning Test 1: Home Page Availability...")
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        print("✅ Home Page Test Passed")

    def test_allowed_ticker(self):
        """Test 2: Ensure a valid ticker (TSLA) generates a prediction."""
        print("\nRunning Test 2: Valid Ticker Prediction (TSLA)...")
        # We send TSLA and expect the result page to contain 'Prediction'
        response = self.app.post('/predict', data={'ticker': 'TSLA'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Prediction', response.data)
        print("✅ Valid Ticker Test Passed")

    def test_invalid_ticker_logic(self):
        """Test 3: Ensure invalid tickers (TWO123) are silently ignored."""
        print("\nRunning Test 3: Invalid Ticker Gatekeeper (TWO123)...")
        response = self.app.post('/predict', data={'ticker': 'TWO123'})
        self.assertEqual(response.status_code, 200)
        # The page should load, but NOT contain the prediction text
        self.assertNotIn(b'Confidence Score', response.data)
        print("✅ Invalid Ticker Gatekeeper Test Passed")

    def test_config_consistency(self):
        """Test 4: Ensure configuration allows our core stocks."""
        print("\nRunning Test 4: Configuration Check...")
        self.assertIn('TSLA', ALLOWED_TICKERS)
        self.assertIn('SPY', ALLOWED_TICKERS)
        print("✅ Configuration Test Passed")

if __name__ == "__main__":
    unittest.main()
```

### **How to Run It**
Now that your main file is named `deploy.py`, make sure to run the tests like this:

```bash
python tests.py
