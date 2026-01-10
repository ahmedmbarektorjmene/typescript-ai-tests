"""
Main evaluator that runs all test suites and generates reports
"""
import torch
import os
import json
import re
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.tokenizer import SimpleByteTokenizer, BytePairTokenizer
from models.mamba2 import Mamba2Model
from evaluation.test_syntax import SyntaxTestSuite
from evaluation.test_programming import ProgrammingTestSuite
from evaluation.test_algorithmic import AlgorithmicTestSuite
from config import EvaluationConfig


class Evaluator:
    """
    Main evaluator for all three test suites
    """
    def __init__(self, config: EvaluationConfig, device: Optional[torch.device] = None):
        self.config = config
        # Auto-detect GPU if available, or use provided device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU for evaluation: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("Using CPU for evaluation")
        else:
            self.device = device
            print(f"Using device: {device}")
        
        # Load tokenizer (same as training)
        self.tokenizer = SimpleByteTokenizer()
        # Try to load saved tokenizer if exists
        tokenizer_path = os.path.join(os.path.dirname(config.checkpoint_path), 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            try:
                self.tokenizer.load(tokenizer_path)
            except:
                pass
        
        # Load model
        self.model = self._load_model(config.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize test suites
        self.syntax_suite = SyntaxTestSuite(config.test_suite_dir)
        self.programming_suite = ProgrammingTestSuite(config.test_suite_dir)
        self.algorithmic_suite = AlgorithmicTestSuite(config.test_suite_dir)
        
        # Results storage
        self.results = {
            'syntax': [],
            'programming': [],
            'algorithmic': []
        }
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine model type from checkpoint, config, or filename
        model_type = checkpoint.get('model_type', None)
        config_dict = checkpoint.get('config', {})
        
        # If model_type not in checkpoint, infer from filename
        if not model_type:
            checkpoint_lower = checkpoint_path.lower()
            if 'rwkv' in checkpoint_lower:
                model_type = 'rwkv_x'
            elif 'xlstm' in checkpoint_lower:
                model_type = 'xlstm'
            elif 'mamba' in checkpoint_lower:
                model_type = 'mamba2'
            else:
                # Try to infer from state_dict keys
                state_dict_keys = list(checkpoint.get('model_state_dict', {}).keys())
                if state_dict_keys:
                    first_key = state_dict_keys[0]
                    if 'time_mix' in first_key or 'channel_mix' in first_key:
                        model_type = 'rwkv_x'
                    elif 'ssm' in first_key or 'A_log' in first_key:
                        model_type = 'mamba2'
                    elif 'cell' in first_key or 'mlstm' in first_key.lower():
                        model_type = 'xlstm'
                    else:
                        model_type = 'mamba2'  # Default
        
        if isinstance(config_dict, str):
            # Config is a string, try to parse or use defaults
            vocab_size = checkpoint.get('vocab_size', self.tokenizer.vocab_size)
            d_model = 512
            n_layers = 6
        else:
            vocab_size = config_dict.get('vocab_size', self.tokenizer.vocab_size)
            d_model = config_dict.get('d_model', 512)
            n_layers = config_dict.get('n_layers', 6)
            mamba2_d_state = config_dict.get('d_state', 16)
            mamba2_d_conv = config_dict.get('d_conv', 4)
            mamba2_expand = config_dict.get('expand', 2)
        
        # Create model based on detected type
            model = Mamba2Model(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                d_state=mamba2_d_state,
                d_conv=mamba2_d_conv,
                expand=mamba2_expand,
            )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def generate_code(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate code completion from prompt
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(self.device)
        
        # Limit input length to avoid issues
        if len(prompt_tokens) > self.config.max_gen_length - 50:
            # Truncate prompt if too long
            input_ids = input_ids[:, -(self.config.max_gen_length - 50):]
            prompt_tokens = prompt_tokens[-(self.config.max_gen_length - 50):]
        
        # Generate
        with torch.no_grad():
            try:
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=min(len(prompt_tokens) + max_length, self.config.max_gen_length),
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
            except Exception as e:
                print(f"Warning: Generation failed: {e}")
                return ""
        
        # Decode
        generated_tokens = generated_ids[0].tolist()
        # Remove prompt tokens
        if len(generated_tokens) > len(prompt_tokens):
            generated_tokens = generated_tokens[len(prompt_tokens):]
        else:
            generated_tokens = []
        
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def extract_completion(self, prompt: str, generated: str) -> str:
        """
        Extract the completion part from generated text
        For syntax tests, we need to extract just the completion
        """
        # Clean generated text - remove invalid Unicode characters
        generated = generated.replace('\ufffd', '').replace('\x00', '').strip()
        
        if not generated:
            return ""
        
        # Find the placeholder position
        if "___" in prompt:
            # For syntax completion, try to extract meaningful completion
            # Look for common patterns after ___
            
            # Try to find completion that looks like code
            # Split by common delimiters
            parts = re.split(r'[;\n\r\t]', generated)
            completion = ""
            
            for part in parts:
                part = part.strip()
                # Look for parts that contain code-like characters
                if part and len(part) > 0:
                    # Check if it contains alphanumeric or common operators
                    if any(c.isalnum() or c in '+-*/=<>()[]{}' for c in part):
                        completion = part
                        # Stop at first semicolon or newline if found
                        if ';' in part:
                            completion = part.split(';')[0] + ';'
                        break
            
            # If no good completion found, take first non-empty line
            if not completion:
                lines = generated.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('//'):
                        completion = line
                        break
            
            # Fallback: take first 50 chars
            if not completion:
                completion = generated[:50].strip()
            
            # Remove any remaining invalid characters
            completion = ''.join(c for c in completion if ord(c) < 0x10000 and c.isprintable())
            return completion
        else:
            # For full function generation, return the generated text
            # Remove invalid characters and non-printable chars
            cleaned = ''.join(c for c in generated if ord(c) < 0x10000 and (c.isprintable() or c in '\n\r\t'))
            return cleaned
    
    def evaluate_syntax(self) -> Dict:
        """Evaluate syntax correctness (Test 1)"""
        print("\n=== Evaluating Syntax Correctness ===")
        tests = self.syntax_suite.get_tests()
        results = []
        correct = 0
        
        for test in tests:
            prompt = test['prompt']
            expected = test.get('expected_completion', '')
            
            # Generate completion
            generated = self.generate_code(prompt, max_length=50)
            completion = self.extract_completion(prompt, generated)
            
            # Filter out completions that are clearly invalid (mostly non-printable or random chars)
            if completion:
                # Check if completion has reasonable code-like content
                printable_ratio = sum(c.isprintable() for c in completion) / len(completion) if completion else 0
                alnum_ratio = sum(c.isalnum() for c in completion) / len(completion) if completion else 0
                
                # If less than 50% printable or less than 20% alphanumeric, it's probably garbage
                if printable_ratio < 0.5 or alnum_ratio < 0.2:
                    completion = ""  # Mark as invalid
            
            # Create complete code
            code = self.syntax_suite.create_test_file(prompt, completion)
            
            # Clean code of invalid Unicode characters
            code = ''.join(c for c in code if ord(c) < 0x10000)
            
            # Validate syntax
            is_valid, error_msg = self.syntax_suite.validate_syntax(code)
            
            result = {
                'test_id': test['id'],
                'prompt': prompt,
                'expected': expected,
                'generated': completion,
                'is_valid': is_valid,
                'error': error_msg
            }
            results.append(result)
            
            if is_valid:
                correct += 1
                print(f"  ✓ {test['id']}: Valid syntax")
            else:
                error_display = error_msg[:100] if error_msg else "Unknown error"
                print(f"  ✗ {test['id']}: Invalid syntax")
                if error_msg:
                    print(f"      Error: {error_display}")
                if completion:
                    # Show cleaned version
                    clean_completion = ''.join(c if c.isprintable() else '?' for c in completion[:50])
                    print(f"      Generated: {clean_completion}...")
                else:
                    print(f"      Generated: (empty or invalid)")
        
        syntax_score = correct / len(tests) if tests else 0.0
        
        return {
            'score': syntax_score,
            'correct': correct,
            'total': len(tests),
            'results': results
        }
    
    def evaluate_programming(self) -> Dict:
        """Evaluate programming correctness (Test 2)"""
        print("\n=== Evaluating Programming Correctness ===")
        tests = self.programming_suite.get_tests()
        results = []
        total_tests = 0
        passed_tests = 0
        
        for test in tests:
            prompt = test['prompt']
            test_cases = test.get('test_cases', [])
            
            # Generate function code
            generated = self.generate_code(prompt, max_length=200)
            
            # Extract function (try to find function definition)
            function_code = generated
            if 'function' not in function_code and 'const' not in function_code:
                # Prepend prompt context
                function_code = prompt + '\n' + generated
            
            test_results = []
            for test_case in test_cases:
                total_tests += 1
                passed, output = self.programming_suite.run_test(function_code, test_case)
                
                if passed:
                    passed_tests += 1
                    print(f"  ✓ {test['id']}: {test_case['description']} - PASSED")
                else:
                    error_display = output[:100] if output else "Test failed"
                    print(f"  ✗ {test['id']}: {test_case['description']} - FAILED")
                    if output and 'PASS' not in output:
                        print(f"      Output: {error_display[:80]}...")
                
                test_results.append({
                    'description': test_case['description'],
                    'passed': passed,
                    'output': output
                })
            
            results.append({
                'test_id': test['id'],
                'prompt': prompt,
                'function_code': function_code,
                'test_results': test_results
            })
        
        programming_score = passed_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            'score': programming_score,
            'passed': passed_tests,
            'total': total_tests,
            'results': results
        }
    
    def evaluate_algorithmic(self) -> Dict:
        """Evaluate algorithmic thinking (Test 3)"""
        print("\n=== Evaluating Algorithmic Thinking ===")
        tests = self.algorithmic_suite.get_tests()
        results = []
        total_score = 0
        max_score = 0
        
        for test in tests:
            prompt = test['prompt']
            test_cases = test.get('test_cases', [])
            efficiency_check = test.get('efficiency_check', '')
            
            # Generate algorithm code
            generated = self.generate_code(prompt, max_length=300)
            
            # Extract function
            # Clean generated text
            generated = ''.join(c for c in generated if ord(c) < 0x10000)
            function_code = generated
            if 'function' not in function_code and 'const' not in function_code:
                function_code = prompt + '\n' + generated
            
            # Run test cases
            all_passed = True
            for test_case in test_cases:
                passed, output = self.algorithmic_suite.run_test(function_code, test_case)
                if not passed:
                    all_passed = False
                    break
            
            # Evaluate efficiency
            if all_passed:
                efficiency_score = self.algorithmic_suite.evaluate_efficiency(
                    function_code, efficiency_check
                )
                score = efficiency_score
            else:
                score = 0
            
            total_score += score
            max_score += 2  # Max score per test is 2
            
            if score == 2:
                print(f"  ✓ {test['id']}: Correct and efficient")
            elif score == 1:
                print(f"  ~ {test['id']}: Partially correct (tests passed but may not be optimal)")
            else:
                print(f"  ✗ {test['id']}: Incorrect (tests failed)")
                # Show first failing test output if available
                if test_cases:
                    for tc in test_cases:
                        passed, output = self.algorithmic_suite.run_test(function_code, tc)
                        if not passed:
                            error_display = output[:80] if output else "Test failed"
                            print(f"      First failure: {error_display}...")
                            break
            
            results.append({
                'test_id': test['id'],
                'prompt': prompt,
                'function_code': function_code,
                'score': score,
                'max_score': 2,
                'efficiency_check': efficiency_check
            })
        
        algorithmic_score = total_score / max_score if max_score > 0 else 0.0
        
        return {
            'score': algorithmic_score,
            'total_score': total_score,
            'max_score': max_score,
            'results': results
        }
    
    def evaluate_all(self) -> Dict:
        """Run all evaluations"""
        print(f"\n{'='*60}")
        print(f"Evaluating model: {self.config.checkpoint_path}")
        print(f"{'='*60}")
        
        # Run all test suites
        syntax_results = self.evaluate_syntax()
        programming_results = self.evaluate_programming()
        algorithmic_results = self.evaluate_algorithmic()
        
        # Calculate composite score
        composite_score = (
            0.2 * syntax_results['score'] +
            0.4 * programming_results['score'] +
            0.4 * algorithmic_results['score']
        )
        
        # Compile results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': self.config.checkpoint_path,
            'syntax': syntax_results,
            'programming': programming_results,
            'algorithmic': algorithmic_results,
            'composite_score': composite_score,
            'scores': {
                'syntax': syntax_results['score'],
                'programming': programming_results['score'],
                'algorithmic': algorithmic_results['score'],
                'composite': composite_score
            }
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Syntax Score:        {syntax_results['score']:.2%} ({syntax_results['correct']}/{syntax_results['total']})")
        print(f"Programming Score:   {programming_results['score']:.2%} ({programming_results['passed']}/{programming_results['total']})")
        print(f"Algorithmic Score:   {algorithmic_results['score']:.2%} ({algorithmic_results['total_score']}/{algorithmic_results['max_score']})")
        print(f"Composite Score:     {composite_score:.2%}")
        print(f"{'='*60}\n")
        
        return final_results
    
    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save evaluation results to file"""
        os.makedirs(self.config.results_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(self.config.checkpoint_path).replace('.pt', '')
            filename = f"results_{model_name}_{timestamp}.json"
        
        filepath = os.path.join(self.config.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
