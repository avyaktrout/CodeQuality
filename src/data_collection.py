"""
Data Collection Module

This module collects Python functions from GitHub to create a dataset for bug prediction.

Data Sources:
1. Buggy Functions: Extracted from bug fix commits (BEFORE version)
2. Clean Functions: From popular, well-tested repositories

GitHub API Rate Limits:
- Authenticated: 5,000 requests per hour
- Unauthenticated: 60 requests per hour (DON'T USE)
- This script implements rate limiting, retry logic, and checkpointing

How to Get a GitHub Token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "Code Quality NN"
4. Check scope: public_repo
5. Generate and copy the token
6. Create .env file with: GITHUB_TOKEN=your_token_here

Why Collect Both Buggy and Clean Code:
- Supervised learning needs labeled examples of both classes
- Buggy: Real bugs from commit history (ground truth)
- Clean: Well-tested popular code (assumed to have fewer bugs)
- Balanced dataset prevents model bias
"""

import ast
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import re

import pandas as pd
import yaml
from dotenv import load_dotenv
from github import Github, GithubException, RateLimitExceededException, Auth
from tqdm import tqdm


@dataclass
class FunctionData:
    """Data structure for storing function information"""
    code: str                  # Function source code
    has_bug: int              # 1 for buggy, 0 for clean
    repo: str                 # Repository name (e.g., "user/repo")
    commit_sha: Optional[str] # Commit hash (for buggy functions)
    file_path: str            # Path to file in repo
    function_name: str        # Name of the function
    stars: int                # Repository star count
    lines_of_code: int        # Number of lines in function


class GitHubCollector:
    """
    Collects Python functions from GitHub for bug prediction dataset.

    Features:
    - Async data collection with rate limit handling
    - Checkpoint/resume capability
    - Progress tracking with tqdm
    - Extensive error handling and logging
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the GitHub collector.

        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load GitHub token from environment
        load_dotenv()
        github_token = os.getenv('GITHUB_TOKEN')

        if not github_token:
            raise ValueError(
                "GitHub token not found! Please:\n"
                "1. Create a personal access token at https://github.com/settings/tokens\n"
                "2. Create a .env file with: GITHUB_TOKEN=your_token_here"
            )

        # Initialize GitHub API client (using new Auth method)
        self.github = Github(auth=Auth.Token(github_token))

        # Verify authentication
        try:
            user = self.github.get_user()
            print(f"Authenticated as: {user.login}")
            self._print_rate_limit()
        except Exception as e:
            raise ValueError(
                f"GitHub authentication failed: {e}\n\n"
                "This usually means your token is invalid, expired, or revoked.\n"
                "Please generate a new token:\n"
                "1. Go to: https://github.com/settings/tokens\n"
                "2. Click 'Generate new token' -> 'Generate new token (classic)'\n"
                "3. Name it: 'Code Quality NN'\n"
                "4. Check scope: 'public_repo'\n"
                "5. Click 'Generate token' and copy it\n"
                "6. Update your .env file with the new token"
            )

        # Extract configuration
        self.data_config = self.config['data_collection']
        self.target_buggy = self.data_config['target_buggy']
        self.target_clean = self.data_config['target_clean']
        self.min_lines = self.data_config['min_lines']
        self.max_lines = self.data_config['max_lines']
        self.min_tokens = self.data_config['min_tokens']
        self.checkpoint_interval = self.data_config['checkpoint_interval']
        self.max_retries = self.data_config['max_retries']

        # Storage
        self.collected_functions: List[FunctionData] = []
        self.seen_code_hashes: Set[str] = set()  # Prevent duplicates

        # Paths
        self.data_dir = Path(self.config['paths']['data_raw'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.data_dir / "collection_checkpoint.json"
        self.output_file = self.data_dir / "functions.csv"

    def _print_rate_limit(self):
        """Print current GitHub API rate limit status"""
        try:
            rate_limit = self.github.get_rate_limit()
            # Handle different PyGithub versions
            if hasattr(rate_limit, 'core'):
                core = rate_limit.core
                search = rate_limit.search
            else:
                # Newer PyGithub versions
                core = rate_limit.rate
                search = getattr(rate_limit, 'search', core)

            print(f"\nGitHub API Rate Limits:")
            print(f"  Core: {core.remaining}/{core.limit} (resets at {core.reset})")
            if hasattr(search, 'remaining'):
                print(f"  Search: {search.remaining}/{search.limit} (resets at {search.reset})")
        except Exception as e:
            print(f"\nCould not fetch rate limits: {e}")

    def _check_rate_limit(self, operation: str = "core"):
        """
        Check rate limit and sleep if necessary.

        Args:
            operation: Type of operation ('core' or 'search')
        """
        try:
            rate_limit = self.github.get_rate_limit()

            # Handle different PyGithub versions
            if hasattr(rate_limit, 'core'):
                limit = rate_limit.core if operation == "core" else rate_limit.search
            else:
                # Newer PyGithub versions use 'rate' instead of 'core'
                limit = rate_limit.rate

            threshold = self.data_config['rate_limit_threshold']

            if limit.remaining < threshold:
                # Calculate sleep time
                reset_time = limit.reset
                sleep_seconds = (reset_time - datetime.now()).total_seconds()
                sleep_seconds += self.data_config['rate_limit_buffer']

                if sleep_seconds > 0:
                    print(f"\nRate limit low ({limit.remaining} remaining). "
                          f"Sleeping for {sleep_seconds:.0f} seconds...")
                    time.sleep(sleep_seconds)
        except Exception as e:
            # If we can't check rate limit, continue anyway
            print(f"Warning: Could not check rate limit: {e}")

    def _hash_code(self, code: str) -> str:
        """
        Create a hash of code for duplicate detection.

        Args:
            code: Source code string

        Returns:
            Hash string
        """
        # Normalize whitespace for better duplicate detection
        normalized = ' '.join(code.split())
        return str(hash(normalized))

    def _is_valid_function(self, code: str) -> bool:
        """
        Check if function meets quality criteria.

        Args:
            code: Function source code

        Returns:
            True if function is valid, False otherwise
        """
        lines = code.split('\n')
        num_lines = len([l for l in lines if l.strip()])  # Non-empty lines

        # Check line count
        if num_lines < self.min_lines or num_lines > self.max_lines:
            return False

        # Check token count (simple approximation)
        tokens = code.split()
        if len(tokens) < self.min_tokens:
            return False

        # Check if it's a duplicate
        code_hash = self._hash_code(code)
        if code_hash in self.seen_code_hashes:
            return False

        return True

    def parse_functions(self, file_content: str, file_path: str) -> List[Dict]:
        """
        Extract function definitions from Python source code.

        Uses Python's AST (Abstract Syntax Tree) to parse code and extract
        function definitions. Handles syntax errors gracefully.

        Args:
            file_content: Python source code
            file_path: Path to the file (for error reporting)

        Returns:
            List of function dictionaries with metadata
        """
        try:
            tree = ast.parse(file_content)
        except SyntaxError as e:
            # This is expected for buggy code
            return []
        except Exception as e:
            print(f"Unexpected parsing error in {file_path}: {e}")
            return []

        functions = []

        # Walk the AST to find function definitions
        for node in ast.walk(tree):
            # Look for function definitions (top-level and methods)
            if isinstance(node, ast.FunctionDef):
                # Skip nested functions (check if parent is FunctionDef)
                if self._is_nested_function(node, tree):
                    continue

                # Skip certain decorators
                if self._has_excluded_decorator(node):
                    continue

                try:
                    # Extract function source code
                    func_source = ast.get_source_segment(file_content, node)

                    if func_source and self._is_valid_function(func_source):
                        # Extract parameter names
                        params = [arg.arg for arg in node.args.args]

                        functions.append({
                            'name': node.name,
                            'code': func_source,
                            'lineno': node.lineno,
                            'end_lineno': node.end_lineno,
                            'params': params,
                            'has_decorator': len(node.decorator_list) > 0
                        })
                except Exception as e:
                    # Skip functions we can't extract
                    continue

        return functions

    def _is_nested_function(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """
        Check if a function is nested inside another function.

        Args:
            func_node: Function AST node to check
            tree: Full AST tree

        Returns:
            True if nested, False if top-level or class method
        """
        # Simple heuristic: check if column offset is > 0
        # Top-level functions and methods start at column 0 or 4
        return func_node.col_offset > 4

    def _has_excluded_decorator(self, func_node: ast.FunctionDef) -> bool:
        """
        Check if function has decorators we want to exclude.

        Args:
            func_node: Function AST node

        Returns:
            True if has excluded decorator
        """
        excluded = ['property', 'abstractmethod', 'abc.abstractmethod']

        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in excluded:
                    return True
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in excluded:
                    return True

        return False

    def match_functions(self,
                       before_funcs: List[Dict],
                       after_funcs: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """
        Match functions between before and after versions of a file.

        Matching algorithm:
        1. Function name must match exactly
        2. Parameters must overlap >= 70%
        3. Line position must be within ±5 lines
        4. Code must actually be different (not just whitespace)

        Args:
            before_funcs: Functions from before commit
            after_funcs: Functions from after commit

        Returns:
            List of (before, after) function pairs
        """
        matches = []

        for before in before_funcs:
            best_match = None
            best_score = 0

            for after in after_funcs:
                # 1. Name must match
                if before['name'] != after['name']:
                    continue

                # 2. Calculate parameter overlap
                before_params = set(before['params'])
                after_params = set(after['params'])

                if len(before_params) == 0 and len(after_params) == 0:
                    param_overlap = 1.0
                else:
                    overlap = len(before_params & after_params)
                    total = max(len(before_params), len(after_params))
                    param_overlap = overlap / total if total > 0 else 0

                if param_overlap < 0.7:
                    continue

                # 3. Check line position similarity
                line_diff = abs(before['lineno'] - after['lineno'])
                if line_diff > 5:
                    continue

                # Calculate match score
                score = param_overlap - (line_diff * 0.01)

                if score > best_score:
                    best_score = score
                    best_match = after

            if best_match:
                # 4. Verify code actually changed
                if self._code_actually_changed(before['code'], best_match['code']):
                    matches.append((before, best_match))

        return matches

    def _code_actually_changed(self, before: str, after: str) -> bool:
        """
        Check if code actually changed (not just whitespace/comments).

        Args:
            before: Code before change
            after: Code after change

        Returns:
            True if substantive change occurred
        """
        # Normalize: remove comments, normalize whitespace
        def normalize(code):
            # Remove comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Normalize whitespace
            code = ' '.join(code.split())
            return code

        return normalize(before) != normalize(after)

    async def collect_buggy_functions(self, target: int = 5000) -> List[FunctionData]:
        """
        Collect buggy functions from GitHub bug fix commits.

        Strategy:
        1. Search commits with bug fix keywords
        2. For each commit, get parent (before) and current (after) versions
        3. Extract Python files that changed
        4. Parse functions from both versions
        5. Match modified functions
        6. Store BEFORE version as buggy

        Args:
            target: Number of buggy functions to collect

        Returns:
            List of FunctionData objects
        """
        print(f"\n{'='*70}")
        print(f"COLLECTING BUGGY FUNCTIONS (Target: {target})")
        print(f"{'='*70}\n")

        buggy_functions = []
        keywords = self.data_config['bug_keywords']
        min_stars = self.data_config['min_stars_buggy']

        # Build search query
        # Example: "fix bug language:Python stars:>100"
        for keyword in keywords:
            if len(buggy_functions) >= target:
                break

            query = f"{keyword} language:Python stars:>{min_stars}"
            print(f"Searching commits: {query}")

            try:
                self._check_rate_limit("search")
                commits = self.github.search_commits(
                    query=query,
                    sort='committer-date',
                    order='desc'
                )

                # Progress bar
                pbar = tqdm(total=target - len(buggy_functions),
                           desc=f"Collecting from '{keyword}'")

                for commit in commits:
                    if len(buggy_functions) >= target:
                        break

                    try:
                        # Rate limit check
                        self._check_rate_limit("core")

                        # Process commit
                        funcs = self._process_bug_fix_commit(commit)
                        buggy_functions.extend(funcs)

                        pbar.update(len(funcs))

                        # Checkpoint periodically
                        if len(buggy_functions) % self.checkpoint_interval == 0:
                            self._save_checkpoint(buggy_functions, [])

                    except GithubException as e:
                        if e.status == 403:  # Rate limit
                            print(f"\nRate limit hit. Waiting...")
                            time.sleep(60)
                        continue
                    except Exception as e:
                        # Skip problematic commits
                        continue

                pbar.close()

            except GithubException as e:
                print(f"Search failed: {e}")
                continue

        print(f"\nCollected {len(buggy_functions)} buggy functions")
        return buggy_functions

    def _process_bug_fix_commit(self, commit) -> List[FunctionData]:
        """
        Process a single bug fix commit to extract buggy functions.

        Args:
            commit: GitHub commit object

        Returns:
            List of FunctionData from this commit
        """
        functions = []

        try:
            repo = commit.repository

            # Skip commits without parent (initial commit)
            if not commit.parents:
                return []

            parent_commit = commit.parents[0]

            # Get modified Python files
            for file in commit.files:
                if not file.filename.endswith('.py'):
                    continue

                # Skip test files (we want real code bugs)
                if 'test' in file.filename.lower():
                    continue

                try:
                    # Get file content at both commits
                    before_content = repo.get_contents(
                        file.filename,
                        ref=parent_commit.sha
                    ).decoded_content.decode('utf-8')

                    after_content = repo.get_contents(
                        file.filename,
                        ref=commit.sha
                    ).decoded_content.decode('utf-8')

                except Exception:
                    # File might not exist in parent (new file) or deleted
                    continue

                # Parse functions
                before_funcs = self.parse_functions(before_content, file.filename)
                after_funcs = self.parse_functions(after_content, file.filename)

                # Match modified functions
                matches = self.match_functions(before_funcs, after_funcs)

                # Store BEFORE versions as buggy
                for before, after in matches:
                    code_hash = self._hash_code(before['code'])
                    if code_hash not in self.seen_code_hashes:
                        self.seen_code_hashes.add(code_hash)

                        functions.append(FunctionData(
                            code=before['code'],
                            has_bug=1,
                            repo=repo.full_name,
                            commit_sha=commit.sha,
                            file_path=file.filename,
                            function_name=before['name'],
                            stars=repo.stargazers_count,
                            lines_of_code=len(before['code'].split('\n'))
                        ))

        except Exception as e:
            # Skip problematic commits
            pass

        return functions

    async def collect_clean_functions(self, target: int = 5000) -> List[FunctionData]:
        """
        Collect clean functions from popular, well-tested repositories.

        Strategy:
        1. Use curated list of high-quality repos (pandas, requests, etc.)
        2. Fetch main branch content
        3. Extract functions from core modules (not tests)
        4. Assume these functions are less likely to have bugs

        Assumption Limitations:
        - Not 100% guaranteed bug-free
        - Addresses educational purpose
        - Real ML systems need manual expert labeling

        Args:
            target: Number of clean functions to collect

        Returns:
            List of FunctionData objects
        """
        print(f"\n{'='*70}")
        print(f"COLLECTING CLEAN FUNCTIONS (Target: {target})")
        print(f"{'='*70}\n")

        clean_functions = []
        clean_repos = self.data_config['clean_repos']

        pbar = tqdm(total=target, desc="Collecting clean functions")

        for repo_name in clean_repos:
            if len(clean_functions) >= target:
                break

            try:
                self._check_rate_limit("core")

                print(f"\nProcessing repo: {repo_name}")
                repo = self.github.get_repo(repo_name)

                # Get Python files from main branch
                contents = repo.get_contents("")

                while contents:
                    file_content = contents.pop(0)

                    if file_content.type == "dir":
                        # Skip test directories
                        if 'test' in file_content.path.lower():
                            continue
                        # Add directory contents to queue
                        contents.extend(repo.get_contents(file_content.path))

                    elif file_content.name.endswith('.py'):
                        # Skip test files
                        if 'test' in file_content.path.lower():
                            continue

                        try:
                            # Get file content
                            code = file_content.decoded_content.decode('utf-8')

                            # Parse functions
                            functions = self.parse_functions(code, file_content.path)

                            # Store as clean functions
                            for func in functions:
                                if len(clean_functions) >= target:
                                    break

                                code_hash = self._hash_code(func['code'])
                                if code_hash not in self.seen_code_hashes:
                                    self.seen_code_hashes.add(code_hash)

                                    clean_functions.append(FunctionData(
                                        code=func['code'],
                                        has_bug=0,
                                        repo=repo.full_name,
                                        commit_sha=None,
                                        file_path=file_content.path,
                                        function_name=func['name'],
                                        stars=repo.stargazers_count,
                                        lines_of_code=len(func['code'].split('\n'))
                                    ))

                                    pbar.update(1)

                        except Exception:
                            continue

            except Exception as e:
                print(f"Error processing {repo_name}: {e}")
                continue

        pbar.close()
        print(f"\nCollected {len(clean_functions)} clean functions")
        return clean_functions

    def _save_checkpoint(self, buggy: List[FunctionData], clean: List[FunctionData]):
        """
        Save checkpoint to allow resuming collection.

        Args:
            buggy: List of buggy functions collected so far
            clean: List of clean functions collected so far
        """
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'buggy_count': len(buggy),
            'clean_count': len(clean),
            'buggy_functions': [asdict(f) for f in buggy],
            'clean_functions': [asdict(f) for f in clean]
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"Checkpoint saved: {len(buggy)} buggy + {len(clean)} clean functions")

    def _load_checkpoint(self) -> Tuple[List[FunctionData], List[FunctionData]]:
        """
        Load checkpoint if exists.

        Returns:
            Tuple of (buggy_functions, clean_functions)
        """
        if not self.checkpoint_file.exists():
            return [], []

        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)

            buggy = [FunctionData(**f) for f in checkpoint['buggy_functions']]
            clean = [FunctionData(**f) for f in checkpoint['clean_functions']]

            # Restore seen hashes
            for func in buggy + clean:
                self.seen_code_hashes.add(self._hash_code(func.code))

            print(f"Loaded checkpoint: {len(buggy)} buggy + {len(clean)} clean functions")
            print(f"Timestamp: {checkpoint['timestamp']}")

            return buggy, clean

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return [], []

    def save_to_csv(self, functions: List[FunctionData]):
        """
        Save collected functions to CSV file.

        Args:
            functions: List of FunctionData to save
        """
        # Convert to DataFrame
        df = pd.DataFrame([asdict(f) for f in functions])

        # Shuffle to mix buggy and clean
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save to CSV
        df.to_csv(self.output_file, index=False)

        print(f"\n{'='*70}")
        print(f"DATA SAVED TO: {self.output_file}")
        print(f"{'='*70}")
        print(f"Total functions: {len(df)}")
        print(f"Buggy functions: {df['has_bug'].sum()}")
        print(f"Clean functions: {(1 - df['has_bug']).sum()}")
        print(f"Average LOC: {df['lines_of_code'].mean():.1f}")
        print(f"Unique repositories: {df['repo'].nunique()}")

    async def collect_all(self):
        """
        Main collection pipeline.

        Steps:
        1. Load checkpoint if exists
        2. Collect buggy functions (from bug fix commits)
        3. Collect clean functions (from popular repos)
        4. Save to CSV
        """
        print(f"\n{'='*70}")
        print(f"CODE QUALITY NN - DATA COLLECTION")
        print(f"{'='*70}\n")

        # Load checkpoint
        buggy_funcs, clean_funcs = self._load_checkpoint()

        # Collect buggy functions
        if len(buggy_funcs) < self.target_buggy:
            remaining = self.target_buggy - len(buggy_funcs)
            new_buggy = await self.collect_buggy_functions(remaining)
            buggy_funcs.extend(new_buggy)
            self._save_checkpoint(buggy_funcs, clean_funcs)

        # Collect clean functions
        if len(clean_funcs) < self.target_clean:
            remaining = self.target_clean - len(clean_funcs)
            new_clean = await self.collect_clean_functions(remaining)
            clean_funcs.extend(new_clean)
            self._save_checkpoint(buggy_funcs, clean_funcs)

        # Combine and save
        all_functions = buggy_funcs + clean_funcs
        self.save_to_csv(all_functions)

        # Print rate limit status
        self._print_rate_limit()

        print("\nData collection complete!")


# Main execution
if __name__ == "__main__":
    # Create collector
    collector = GitHubCollector(config_path="config.yaml")

    # Run collection
    asyncio.run(collector.collect_all())
