"""
Test Suite 1: Syntax Correctness
Tests model's ability to complete or repair TypeScript code syntax
"""
import os
import json
from typing import List, Dict, Tuple


class SyntaxTestSuite:
    """
    Test suite for syntax correctness evaluation
    """
    def __init__(self, test_suite_dir: str = "evaluation/test_suites"):
        self.test_suite_dir = test_suite_dir
        os.makedirs(test_suite_dir, exist_ok=True)
        self.tests = self._load_tests()
    
    def _load_tests(self) -> List[Dict]:
        """Load syntax tests from file or create default tests"""
        test_file = os.path.join(self.test_suite_dir, "syntax_tests.json")
        
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                return json.load(f)
        
        # Default test cases
        default_tests = [
            {
                "id": "syntax_001",
                "prompt": "function sum(a: number, b: number): number {\n    return a ___;\n}",
                "expected_completion": "+ b;",
                "description": "Complete arithmetic expression"
            },
            {
                "id": "syntax_002",
                "prompt": "interface User {\n    name: string;\n    age: ___;\n}",
                "expected_completion": "number",
                "description": "Complete type annotation"
            },
            {
                "id": "syntax_003",
                "prompt": "const arr = [1, 2, 3];\narr.___((x) => x * 2);",
                "expected_completion": ".map",
                "description": "Complete array method"
            },
            {
                "id": "syntax_004",
                "prompt": "class MyClass {\n    constructor(public name: string) {}\n    greet() {\n        return `Hello, ${this.___}`;\n    }\n}",
                "expected_completion": "name",
                "description": "Complete property access"
            },
            {
                "id": "syntax_005",
                "prompt": "async function fetchData() {\n    const response = await fetch('https://api.example.com');\n    const data = await response.___();\n}",
                "expected_completion": ".json()",
                "description": "Complete async method call"
            },
            {
                "id": "syntax_006",
                "prompt": "type Status = 'pending' | 'completed' | ___;",
                "expected_completion": "'failed'",
                "description": "Complete union type"
            },
            {
                "id": "syntax_007",
                "prompt": "const result = numbers.filter(n => n ___ 0);",
                "expected_completion": ">",
                "description": "Complete comparison operator"
            },
            {
                "id": "syntax_008",
                "prompt": "try {\n    riskyOperation();\n} catch (error) {\n    console.___(error);\n}",
                "expected_completion": ".error",
                "description": "Complete console method"
            },
            {
                "id": "syntax_009",
                "prompt": "const config: { apiKey: string; timeout: ___ } = {\n    apiKey: '123',\n    timeout: 5000\n};",
                "expected_completion": "number",
                "description": "Complete object type annotation"
            },
            {
                "id": "syntax_010",
                "prompt": "function process<T>(items: T[]): T[] {\n    return items.map(item => transform(item));\n}\nconst numbers = process<___>([1, 2, 3]);",
                "expected_completion": "number",
                "description": "Complete generic type parameter"
            }
        ]
        
        # Save default tests
        with open(test_file, 'w') as f:
            json.dump(default_tests, f, indent=2)
        
        return default_tests
    
    def get_tests(self) -> List[Dict]:
        """Get all test cases"""
        return self.tests
    
    def create_test_file(self, prompt: str, completion: str) -> str:
        """
        Create a complete TypeScript file from prompt and completion
        """
        # Replace placeholder with completion
        code = prompt.replace("___", completion)
        return code
    
    def validate_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate TypeScript syntax using external compiler
        Returns (is_valid, error_message)
        """
        import tempfile
        import subprocess
        
        # Create temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False, encoding='utf-8', errors='replace') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Try to compile with bun x tsc (if available)
            # Fallback: basic syntax checking
            result = subprocess.run(
                ['bun', 'x', 'tsc', '--noEmit', temp_path],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                error_msg = result.stderr or result.stdout or "Compilation failed"
                # Clean up error message
                error_msg = error_msg.replace(temp_path, '<file>')
                return False, error_msg
        except FileNotFoundError:
            # bun not available, try global tsc
            try:
                result = subprocess.run(
                    ['tsc', '--noEmit', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    encoding='utf-8',
                    errors='replace'
                )
                if result.returncode == 0:
                    return True, ""
                else:
                    return False, result.stderr or result.stdout or "Compilation failed"
            except:
                # Basic validation fallback
            # Check for common syntax errors
            errors = []
            if code.count('{') != code.count('}'):
                errors.append("Mismatched braces")
            if code.count('(') != code.count(')'):
                errors.append("Mismatched parentheses")
            if code.count('[') != code.count(']'):
                errors.append("Mismatched brackets")
            
            if errors:
                return False, "; ".join(errors)
            # Basic check passed
            return True, ""
        except Exception as e:
            return False, f"Validation error: {str(e)}"
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
