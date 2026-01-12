"""
Test Suite 2: Programming Correctness
Tests model's ability to generate working functions with test cases
"""
import os
import json
from typing import List, Dict, Tuple


class ProgrammingTestSuite:
    """
    Test suite for programming correctness evaluation
    """
    def __init__(self, test_suite_dir: str = "evaluation/test_suites"):
        self.test_suite_dir = test_suite_dir
        os.makedirs(test_suite_dir, exist_ok=True)
        self.tests = self._load_tests()
    
    def _load_tests(self) -> List[Dict]:
        """Load programming tests from file or create default tests"""
        test_file = os.path.join(self.test_suite_dir, "programming_tests.json")
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                return json.load(f)
        
        # Default test cases
        default_tests = [
            {
                "id": "prog_001",
                "prompt": "Write a TypeScript function `fizzBuzz(n: number): string[]` that returns the FizzBuzz sequence up to n.",
                "test_cases": [
                    {
                        "input": "fizzBuzz(5)",
                        "expected": '["1","2","Fizz","4","Buzz"]',
                        "description": "Basic FizzBuzz test"
                    },
                    {
                        "input": "fizzBuzz(15)",
                        "expected": '["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]',
                        "description": "Extended FizzBuzz test"
                    }
                ]
            },
            {
                "id": "prog_002",
                "prompt": "Write a TypeScript function `reverseString(s: string): string` that reverses a string.",
                "test_cases": [
                    {
                        "input": "reverseString('hello')",
                        "expected": "'olleh'",
                        "description": "Basic string reversal"
                    },
                    {
                        "input": "reverseString('')",
                        "expected": "''",
                        "description": "Empty string"
                    }
                ]
            },
            {
                "id": "prog_003",
                "prompt": "Write a TypeScript function `isPalindrome(s: string): boolean` that checks if a string is a palindrome.",
                "test_cases": [
                    {
                        "input": "isPalindrome('racecar')",
                        "expected": "true",
                        "description": "Valid palindrome"
                    },
                    {
                        "input": "isPalindrome('hello')",
                        "expected": "false",
                        "description": "Not a palindrome"
                    }
                ]
            },
            {
                "id": "prog_004",
                "prompt": "Write a TypeScript function `factorial(n: number): number` that calculates the factorial of n.",
                "test_cases": [
                    {
                        "input": "factorial(5)",
                        "expected": "120",
                        "description": "Factorial of 5"
                    },
                    {
                        "input": "factorial(0)",
                        "expected": "1",
                        "description": "Factorial of 0"
                    }
                ]
            },
            {
                "id": "prog_005",
                "prompt": "Write a TypeScript function `fibonacci(n: number): number` that returns the nth Fibonacci number.",
                "test_cases": [
                    {
                        "input": "fibonacci(7)",
                        "expected": "13",
                        "description": "7th Fibonacci number"
                    },
                    {
                        "input": "fibonacci(0)",
                        "expected": "0",
                        "description": "0th Fibonacci number"
                    }
                ]
            },
            {
                "id": "prog_006",
                "prompt": "Write a TypeScript function `maxArray(arr: number[]): number` that returns the maximum value in an array.",
                "test_cases": [
                    {
                        "input": "maxArray([1, 5, 3, 9, 2])",
                        "expected": "9",
                        "description": "Find maximum"
                    },
                    {
                        "input": "maxArray([-10, -5, -1])",
                        "expected": "-1",
                        "description": "Maximum with negatives"
                    }
                ]
            },
            {
                "id": "prog_007",
                "prompt": "Write a TypeScript function `countWords(s: string): number` that counts the number of words in a string.",
                "test_cases": [
                    {
                        "input": "countWords('hello world')",
                        "expected": "2",
                        "description": "Two words"
                    },
                    {
                        "input": "countWords('')",
                        "expected": "0",
                        "description": "Empty string"
                    }
                ]
            },
            {
                "id": "prog_008",
                "prompt": "Write a TypeScript function `isPrime(n: number): boolean` that checks if a number is prime.",
                "test_cases": [
                    {
                        "input": "isPrime(7)",
                        "expected": "true",
                        "description": "Prime number"
                    },
                    {
                        "input": "isPrime(10)",
                        "expected": "false",
                        "description": "Composite number"
                    }
                ]
            }
        ]
        
        # Save default tests
        with open(test_file, 'w') as f:
            json.dump(default_tests, f, indent=2)
        
        return default_tests
    
    def get_tests(self) -> List[Dict]:
        """Get all test cases"""
        return self.tests
    
    def create_test_harness(self, function_code: str, test_case: Dict) -> str:
        """
        Create a test harness for running a test case
        """
        test_code = f"""
{function_code}

// Test case
const result = {test_case['input']};
const expected = {test_case['expected']};

// Simple comparison (works for primitives and arrays)
function arrayEquals(a: any, b: any): boolean {{
    if (a === b) return true;
    if (Array.isArray(a) && Array.isArray(b)) {{
        if (a.length !== b.length) return false;
        return a.every((val, idx) => val === b[idx]);
    }}
    return false;
}}

const passed = arrayEquals(result, expected);
console.log(passed ? 'PASS' : 'FAIL');
if (!passed) {{
    console.log('Expected:', expected);
    console.log('Got:', result);
}}
"""
        return test_code
    
    def run_test(self, function_code: str, test_case: Dict) -> Tuple[bool, str]:
        """
        Run a test case and return (passed, output)
        """
        import tempfile
        import subprocess
        
        test_harness = self.create_test_harness(function_code, test_case)
        
        # Create temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False, encoding='utf-8', errors='replace') as f:
            f.write(test_harness)
            temp_path = f.name
        
        try:
            # Try to run with bun (if available)
            result = subprocess.run(
                ['bun', 'run', temp_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and 'PASS' in result.stdout:
                return True, result.stdout
            else:
                # If bun failed but exists, return its output
                return False, result.stderr or result.stdout
        except FileNotFoundError:
            # bun not available, try nodes/tsx
            try:
                result = subprocess.run(
                    ['bunx', '--bun', 'tsx', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and 'PASS' in result.stdout:
                    return True, result.stdout
                else:
                    return False, result.stderr or result.stdout
            except:
                return False, "No TypeScript runtime available (bun required)"
        except Exception as e:
            return False, str(e)
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
