import unittest
import json
import os
from synthetic_data.pipeline import SyntheticDataPipeline
from unittest.mock import MagicMock, patch

class TestSyntheticDataPipeline(unittest.TestCase):
    
    @patch('synthetic_data.pipeline.cohere.ClientV2')
    def test_pipeline_structure(self, mock_client):
        # Setup mock response
        mock_instance = mock_client.return_value
        
        # Mock scenario generation response
        scenario_block = MagicMock()
        scenario_block.type = "text"
        scenario_block.text = json.dumps({
            "scenario_description": "Test scenario",
            "user_profile": "Test user",
            "key_signals_to_include": ["signal1"],
            "distractor_signals": [],
            "suggested_turn_breakdown": "Test breakdown"
        })
        mock_scenario_response = MagicMock()
        mock_scenario_response.message.content = [scenario_block]
        
        # Mock conversation generation response
        conv_block = MagicMock()
        conv_block.type = "text"
        conv_block.text = json.dumps({
            "scenario_id": "test_id_001",
            "conversation": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"}
            ],
            "labels": {
                "categories": ["company.brand_core"],
                "persistence_horizon": "long",
                "memory_scope": "company",
                "rationale": "test rationale"
            },
            "metadata": {
                "scenario_type": "test",
                "primary_category": "company.brand_core",
                "distractor_present": False,
                "turn_count": 2,
                "signals_present": ["signal1"]
            }
        })
        mock_conv_response = MagicMock()
        mock_conv_response.message.content = [conv_block]
        
        mock_instance.chat.side_effect = [mock_scenario_response, mock_conv_response]
        
        # Run pipeline with dummy key
        pipeline = SyntheticDataPipeline(api_key="dummy_key")
        results = pipeline.run_batch(count=1)
        
        self.assertEqual(len(results), 1)
        self.assertIn('scenario_id', results[0])
        self.assertIn('conversation', results[0])
        self.assertIn('labels', results[0])
        self.assertEqual(results[0]['labels']['categories'], ["company.brand_core"])

    def test_conversation_structure(self):
        # Validate structure of a sample output
        sample_data = {
            "scenario_id": "brand_core_test_001",
            "conversation": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"}
            ],
            "labels": {
                "categories": ["company.brand_core"],
                "persistence_horizon": "long",
                "memory_scope": "company",
                "rationale": "test"
            },
            "metadata": {
                "scenario_type": "test",
                "primary_category": "company.brand_core",
                "distractor_present": False,
                "turn_count": 2,
                "signals_present": []
            }
        }
        
        # Check required fields
        required_fields = ['scenario_id', 'conversation', 'labels', 'metadata']
        for field in required_fields:
            self.assertIn(field, sample_data)
            
        # Check label structure
        label_fields = ['categories', 'persistence_horizon', 'memory_scope', 'rationale']
        for field in label_fields:
            self.assertIn(field, sample_data['labels'])

if __name__ == '__main__':
    unittest.main()

