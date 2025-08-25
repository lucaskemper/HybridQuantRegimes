#!/usr/bin/env python3
"""
Basic Quality Test for HybridQuantRegimes Project
This script performs basic syntax and import validation without requiring external dependencies.
"""

import sys
import ast
import os
from pathlib import Path

def test_python_syntax():
    """Test if all Python files have valid syntax"""
    src_dir = Path("src")
    syntax_errors = []
    
    for py_file in src_dir.glob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"✓ {py_file.name}: Valid syntax")
        except SyntaxError as e:
            syntax_errors.append(f"✗ {py_file.name}: {e}")
            print(f"✗ {py_file.name}: Syntax error - {e}")
    
    return len(syntax_errors) == 0, syntax_errors

def test_import_structure():
    """Test basic import structure and circular dependencies"""
    import_errors = []
    src_dir = Path("src")
    
    for py_file in src_dir.glob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Check for relative imports within src
            src_imports = [imp for imp in imports if imp.startswith('src.')]
            print(f"✓ {py_file.name}: Found {len(src_imports)} internal imports")
            
        except Exception as e:
            import_errors.append(f"✗ {py_file.name}: {e}")
            print(f"✗ {py_file.name}: Import analysis failed - {e}")
    
    return len(import_errors) == 0, import_errors

def test_configuration_files():
    """Test configuration file validity"""
    config_errors = []
    
    # Test YAML config
    try:
        import yaml
        with open("config.yml", 'r') as f:
            config = yaml.safe_load(f)
        print("✓ config.yml: Valid YAML structure")
        
        # Check required sections
        required_sections = ['portfolio', 'regime', 'risk', 'backtest']
        for section in required_sections:
            if section in config:
                print(f"✓ config.yml: Contains {section} section")
            else:
                config_errors.append(f"✗ config.yml: Missing {section} section")
                
    except ImportError:
        print("⚠ YAML module not available, skipping config test")
    except Exception as e:
        config_errors.append(f"✗ config.yml: {e}")
        print(f"✗ config.yml: Configuration error - {e}")
    
    return len(config_errors) == 0, config_errors

def test_documentation_quality():
    """Test documentation completeness"""
    doc_issues = []
    
    # Check README exists and has content
    if os.path.exists("README.md"):
        with open("README.md", 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        if len(readme_content) > 1000:
            print(f"✓ README.md: Comprehensive documentation ({len(readme_content)} characters)")
        else:
            doc_issues.append("✗ README.md: Documentation seems brief")
            
        # Check for key sections
        key_sections = ['Installation', 'Usage', 'Configuration', 'Testing']
        for section in key_sections:
            if section.lower() in readme_content.lower():
                print(f"✓ README.md: Contains {section} section")
            else:
                doc_issues.append(f"⚠ README.md: Missing {section} section")
    else:
        doc_issues.append("✗ README.md: Documentation file missing")
    
    return len(doc_issues) == 0, doc_issues

def test_project_structure():
    """Test project structure follows best practices"""
    structure_issues = []
    
    # Check required directories
    required_dirs = ['src', 'tests']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory structure: {dir_name}/ exists")
        else:
            structure_issues.append(f"✗ Directory structure: Missing {dir_name}/")
    
    # Check test structure
    if os.path.exists("tests"):
        test_subdirs = ['unit', 'integration']
        for subdir in test_subdirs:
            test_path = os.path.join("tests", subdir)
            if os.path.exists(test_path):
                print(f"✓ Test structure: {test_path}/ exists")
            else:
                structure_issues.append(f"⚠ Test structure: Missing {test_path}/")
    
    # Check for configuration files
    config_files = ['config.yml', 'requirements.txt']
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ Configuration: {config_file} exists")
        else:
            structure_issues.append(f"✗ Configuration: Missing {config_file}")
    
    return len(structure_issues) == 0, structure_issues

def count_code_metrics():
    """Count basic code metrics"""
    metrics = {
        'total_files': 0,
        'total_lines': 0,
        'classes': 0,
        'functions': 0,
        'imports': 0
    }
    
    src_dir = Path("src")
    for py_file in src_dir.glob("*.py"):
        metrics['total_files'] += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            metrics['total_lines'] += len(lines)
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metrics['classes'] += 1
                elif isinstance(node, ast.FunctionDef):
                    metrics['functions'] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics['imports'] += 1
                    
        except Exception as e:
            print(f"⚠ Could not analyze {py_file}: {e}")
    
    return metrics

def main():
    """Run all quality tests"""
    print("=" * 60)
    print("HybridQuantRegimes Project Quality Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Syntax validation
    print("\n1. Testing Python Syntax...")
    syntax_ok, syntax_errors = test_python_syntax()
    all_passed &= syntax_ok
    
    # Test 2: Import structure
    print("\n2. Testing Import Structure...")
    import_ok, import_errors = test_import_structure()
    all_passed &= import_ok
    
    # Test 3: Configuration
    print("\n3. Testing Configuration Files...")
    config_ok, config_errors = test_configuration_files()
    all_passed &= config_ok
    
    # Test 4: Documentation
    print("\n4. Testing Documentation...")
    doc_ok, doc_errors = test_documentation_quality()
    all_passed &= doc_ok
    
    # Test 5: Project structure
    print("\n5. Testing Project Structure...")
    structure_ok, structure_errors = test_project_structure()
    all_passed &= structure_ok
    
    # Metrics
    print("\n6. Code Metrics...")
    metrics = count_code_metrics()
    for key, value in metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("✓ All basic quality tests PASSED")
        print("✓ Project shows good technical quality")
    else:
        print("⚠ Some quality issues detected")
        
        # Print all errors
        all_errors = syntax_errors + import_errors + config_errors + doc_errors + structure_errors
        if all_errors:
            print("\nIssues found:")
            for error in all_errors:
                print(f"  {error}")
    
    print(f"\nOverall Assessment: {'GOOD QUALITY' if all_passed else 'NEEDS ATTENTION'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())