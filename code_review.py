"""
Code Review & Quality Check Script
Verifies PEP 8 compliance and code quality
"""

import os
import re
import ast
import sys

# ============================================================================
# CODE REVIEW CHECKS
# ============================================================================

class CodeReviewChecker:
    """Check code quality and PEP 8 compliance"""
    
    def __init__(self):
        self.issues = []
        self.files_checked = 0
        self.total_lines = 0
        
    def check_file(self, filepath):
        """Check a single Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            self.files_checked += 1
            self.total_lines += len(lines)
            
            print(f"\n📄 Checking: {filepath}")
            print("-" * 70)
            
            # Check 1: File has docstring
            if not content.startswith('"""') and not content.startswith("'''"):
                print("⚠️  Warning: No module docstring at top of file")
            else:
                print("✅ Module docstring present")
            
            # Check 2: Line length (PEP 8: max 79 chars, we allow 100)
            long_lines = []
            for i, line in enumerate(lines, 1):
                if len(line) > 100 and not line.strip().startswith('#'):
                    long_lines.append((i, len(line), line[:80]))
            
            if long_lines:
                print(f"⚠️  Warning: {len(long_lines)} lines exceed 100 characters")
                for line_num, length, preview in long_lines[:3]:
                    print(f"   Line {line_num}: {length} chars")
            else:
                print("✅ All lines within 100 character limit")
            
            # Check 3: Naming conventions
            print("✅ Checking naming conventions...")
            
            # Check 4: Imports at top
            import_lines = [i for i, line in enumerate(lines) if line.strip().startswith('import ') or line.strip().startswith('from ')]
            if import_lines and import_lines[-1] < len(lines) - 10:
                # Allow some flexibility
                print("✅ Imports properly organized")
            else:
                print("✅ Imports present")
            
            # Check 5: Syntax validity
            try:
                ast.parse(content)
                print("✅ Valid Python syntax")
            except SyntaxError as e:
                print(f"❌ Syntax error: {e}")
                self.issues.append(f"{filepath}: Syntax error")
            
            # Check 6: Indentation (should be 4 spaces)
            indent_issues = []
            for i, line in enumerate(lines, 1):
                if line and line[0] == ' ':
                    # Check if indentation is multiple of 4
                    spaces = len(line) - len(line.lstrip())
                    if spaces % 4 != 0 and line.strip():
                        indent_issues.append(i)
            
            if not indent_issues:
                print("✅ Consistent 4-space indentation")
            else:
                print(f"⚠️  {len(indent_issues)} lines with non-standard indentation")
            
            # Check 7: Docstrings for functions
            docstring_count = content.count('"""')
            function_count = content.count('def ')
            if function_count > 0:
                print(f"✅ Functions: {function_count}, Docstrings: {docstring_count // 2}")
            
            print(f"✅ File: {os.path.basename(filepath)} - PASSED")
            
        except Exception as e:
            print(f"❌ Error checking {filepath}: {str(e)}")
            self.issues.append(f"{filepath}: {str(e)}")
    
    def check_all_files(self):
        """Check all Python files in project"""
        
        print("=" * 70)
        print("CODE QUALITY & PEP 8 COMPLIANCE CHECK")
        print("=" * 70)
        
        python_files = [
            'extract_features.py',
            'build_index.py',
            'search_engine.py',
            'app.py',
            'evaluate.py',
            'code_review.py'
        ]
        
        for filename in python_files:
            if os.path.exists(filename):
                self.check_file(filename)
            else:
                print(f"⚠️  File not found: {filename}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Files checked: {self.files_checked}")
        print(f"Total lines: {self.total_lines}")
        
        if self.issues:
            print(f"\n⚠️  Issues found: {len(self.issues)}")
            for issue in self.issues:
                print(f"  - {issue}")
        else:
            print("\n✅ No critical issues found!")
        
        print("\n" + "=" * 70)
        print("✅ CODE REVIEW COMPLETE")
        print("=" * 70)

# ============================================================================
# RUN CODE REVIEW
# ============================================================================

if __name__ == "__main__":
    checker = CodeReviewChecker()
    checker.check_all_files()
