import adalflow as adal
from adalflow.core.types import Document, List
from typing import Optional
from adalflow.components.data_process import TextSplitter, ToEmbeddings
import os
import subprocess
import json
import tiktoken
import logging
import base64
import glob
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.db import LocalDB
from api.config import configs, DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILES
from api.ollama_patch import OllamaDocumentProcessor
from urllib.parse import urlparse, urlunparse, quote
import requests
from requests.exceptions import RequestException

from api.tools.embedder import get_embedder

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for OpenAI embedding models
MAX_EMBEDDING_TOKENS = 8192

def count_tokens(text: str, embedder_type: str = None, is_ollama_embedder: bool = None) -> int:
    """
    Count the number of tokens in a text string using tiktoken.

    Args:
        text (str): The text to count tokens for.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama', 'bedrock').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.

    Returns:
        int: The number of tokens in the text.
    """
    try:
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        
        # Determine embedder type if not specified
        if embedder_type is None:
            from api.config import get_embedder_type
            embedder_type = get_embedder_type()

        # Choose encoding based on embedder type
        if embedder_type == 'ollama':
            # Ollama typically uses cl100k_base encoding
            encoding = tiktoken.get_encoding("cl100k_base")
        elif embedder_type == 'google':
            # Google uses similar tokenization to GPT models for rough estimation
            encoding = tiktoken.get_encoding("cl100k_base")
        elif embedder_type == 'bedrock':
            # Bedrock embedding models vary; use a common GPT-like encoding for rough estimation
            encoding = tiktoken.get_encoding("cl100k_base")
        else:  # OpenAI or default
            # Use OpenAI embedding model encoding
            encoding = tiktoken.encoding_for_model("text-embedding-3-small")

        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to a simple approximation if tiktoken fails
        logger.warning(f"Error counting tokens with tiktoken: {e}")
        # Rough approximation: 4 characters per token
        return len(text) // 4

def _repo_has_content(path: str) -> bool:
    """Check if a repo directory has actual content files (not just .git and dotfiles)."""
    if not os.path.exists(path):
        return False
    entries = os.listdir(path)
    # A repo with only dotfiles (e.g. .git, .gitignore) is considered empty/broken
    non_dot_entries = [e for e in entries if not e.startswith('.')]
    return len(non_dot_entries) > 0

def download_repo(repo_url: str, local_path: str, repo_type: Optional[str] = None, access_token: Optional[str] = None, branch: Optional[str] = None) -> str:
    """
    Downloads a Git repository (GitHub, GitLab, or Bitbucket) to a specified local path.

    Args:
        repo_type(str): Type of repository
        repo_url (str): The URL of the Git repository to clone.
        local_path (str): The local directory where the repository will be cloned.
        access_token (str, optional): Access token for private repositories.
        branch (str, optional): Branch to clone. Defaults to the repository's default branch.

    Returns:
        str: The output message from the `git` command.
    """
    try:
        # Check if Git is installed
        logger.info(f"Preparing to clone repository to {local_path}")
        subprocess.run(
            ["git", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Check if repository already exists with actual content
        if os.path.exists(local_path) and _repo_has_content(local_path):
            logger.warning(f"Repository already exists at {local_path}. Using existing repository.")
            return f"Using existing repository at {local_path}"
        elif os.path.exists(local_path) and os.listdir(local_path):
            # Directory exists but only has dotfiles (.git, .gitignore) – broken/empty clone
            logger.warning(f"Repository at {local_path} has no content files (only dotfiles). Removing and re-cloning.")
            import shutil
            shutil.rmtree(local_path)

        # Ensure the local path exists
        os.makedirs(local_path, exist_ok=True)

        # Prepare the clone URL and git arguments
        clone_url = repo_url
        extra_git_args: list = []  # additional -c flags prepended before "clone"

        if access_token:
            parsed = urlparse(repo_url)
            encoded_token = quote(access_token, safe='')

            if repo_type == "github":
                # Format: https://{token}@{domain}/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
                logger.info("Using access token for GitHub authentication")

            elif repo_type == "gitlab":
                # Format: https://oauth2:{token}@gitlab.com/owner/repo.git
                clone_url = urlunparse((parsed.scheme, f"oauth2:{encoded_token}@{parsed.netloc}", parsed.path, '', '', ''))
                logger.info("Using access token for GitLab authentication")

            elif repo_type == "bitbucket":
                # Bitbucket Server PAT: use http.extraHeader so Bearer auth works regardless
                # of server version (x-token-auth in the URL only works on newer versions).
                # Convert browse URL (/projects/PROJ/repos/REPO) → SCM clone URL if needed.
                parsed_path = parsed.path
                path_parts = [p for p in parsed_path.split('/') if p]
                if 'projects' in path_parts and 'repos' in path_parts:
                    repos_idx = path_parts.index('repos')
                    project_key = path_parts[repos_idx - 1] if repos_idx > 0 else ''
                    repo_slug = path_parts[repos_idx + 1] if repos_idx + 1 < len(path_parts) else ''
                    if project_key and repo_slug:
                        parsed_path = f"/scm/{project_key}/{repo_slug}.git"
                        logger.info(f"Converted Bitbucket Server browse URL to SCM clone path: {parsed_path}")
                clone_url = urlunparse((parsed.scheme, parsed.netloc, parsed_path, '', '', ''))
                # Pass PAT as Bearer token via HTTP header (works for all Bitbucket Server versions)
                extra_git_args = ["-c", f"http.extraHeader=Authorization: Bearer {access_token}"]
                logger.info("Using Bearer token (http.extraHeader) for Bitbucket authentication")

        # For Bitbucket Server browse URLs without a token, also convert to SCM clone URL
        elif repo_type == "bitbucket" and not access_token:
            parsed = urlparse(repo_url)
            path_parts = [p for p in parsed.path.split('/') if p]
            if 'projects' in path_parts and 'repos' in path_parts:
                repos_idx = path_parts.index('repos')
                project_key = path_parts[repos_idx - 1] if repos_idx > 0 else ''
                repo_slug = path_parts[repos_idx + 1] if repos_idx + 1 < len(path_parts) else ''
                if project_key and repo_slug:
                    clone_path = f"/scm/{project_key}/{repo_slug}.git"
                    clone_url = urlunparse((parsed.scheme, parsed.netloc, clone_path, '', '', ''))
                    logger.info(f"Converted Bitbucket Server browse URL to SCM clone path: {clone_path}")

        # Clone the repository
        branch_desc = f" (branch: {branch})" if branch else " (default branch)"
        logger.info(f"Cloning repository from {repo_url} to {local_path}{branch_desc}")
        # Build the command: git [-c key=value ...] clone --depth=1 --single-branch [--branch <branch>] <url> <path>
        branch_args = ["--branch", branch] if branch else []
        git_cmd = ["git"] + extra_git_args + ["clone", "--depth=1", "--single-branch"] + branch_args + [clone_url, local_path]
        result = subprocess.run(
            git_cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        logger.info("Repository cloned successfully")
        # Log a quick summary of what was cloned
        try:
            top = os.listdir(local_path)
            logger.info(f"Cloned repo top-level ({len(top)} entries): {sorted(top)[:20]}")
        except Exception:
            pass
        return result.stdout.decode("utf-8")

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8')
        # Sanitize error message to remove any tokens (both raw and URL-encoded)
        if access_token:
            # Remove raw token
            error_msg = error_msg.replace(access_token, "***TOKEN***")
            # Also remove URL-encoded token to prevent leaking encoded version
            encoded_token = quote(access_token, safe='')
            error_msg = error_msg.replace(encoded_token, "***TOKEN***")
        raise ValueError(f"Error during cloning: {error_msg}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {str(e)}")

# Alias for backward compatibility
download_github_repo = download_repo

def read_all_documents(path: str, embedder_type: str = None, is_ollama_embedder: bool = None, 
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
    """
    Recursively reads all documents in a directory and its subdirectories.

    Args:
        path (str): The root directory path.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.
        excluded_dirs (List[str], optional): List of directories to exclude from processing.
            Overrides the default configuration if provided.
        excluded_files (List[str], optional): List of file patterns to exclude from processing.
            Overrides the default configuration if provided.
        included_dirs (List[str], optional): List of directories to include exclusively.
            When provided, only files in these directories will be processed.
        included_files (List[str], optional): List of file patterns to include exclusively.
            When provided, only files matching these patterns will be processed.

    Returns:
        list: A list of Document objects with metadata.
    """
    # Handle backward compatibility
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None
    documents = []
    # File extensions to look for, prioritizing code files
    code_extensions = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".vue", ".html", ".css", ".scss", ".less",
                       ".php", ".swift", ".cs", ".rb", ".kt", ".scala", ".dart"]
    doc_extensions = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]

    # Determine filtering mode: inclusion or exclusion
    use_inclusion_mode = (included_dirs is not None and len(included_dirs) > 0) or (included_files is not None and len(included_files) > 0)

    if use_inclusion_mode:
        # Inclusion mode: only process specified directories and files
        final_included_dirs = set(included_dirs) if included_dirs else set()
        final_included_files = set(included_files) if included_files else set()

        logger.info(f"Using inclusion mode")
        logger.info(f"Included directories: {list(final_included_dirs)}")
        logger.info(f"Included files: {list(final_included_files)}")

        # Convert to lists for processing
        included_dirs = list(final_included_dirs)
        included_files = list(final_included_files)
        excluded_dirs = []
        excluded_files = []
    else:
        # Exclusion mode: use default exclusions plus any additional ones
        final_excluded_dirs = set(DEFAULT_EXCLUDED_DIRS)
        final_excluded_files = set(DEFAULT_EXCLUDED_FILES)

        # Add any additional excluded directories from config
        if "file_filters" in configs and "excluded_dirs" in configs["file_filters"]:
            final_excluded_dirs.update(configs["file_filters"]["excluded_dirs"])

        # Add any additional excluded files from config
        if "file_filters" in configs and "excluded_files" in configs["file_filters"]:
            final_excluded_files.update(configs["file_filters"]["excluded_files"])

        # Add any explicitly provided excluded directories and files
        if excluded_dirs is not None:
            final_excluded_dirs.update(excluded_dirs)

        if excluded_files is not None:
            final_excluded_files.update(excluded_files)

        # Convert back to lists for compatibility
        excluded_dirs = list(final_excluded_dirs)
        excluded_files = list(final_excluded_files)
        included_dirs = []
        included_files = []

        logger.info(f"Using exclusion mode")
        logger.info(f"Excluded directories: {excluded_dirs}")
        logger.info(f"Excluded files: {excluded_files}")

    logger.info(f"Reading documents from {path}")

    # Log what actually exists in the repo root for diagnostics
    try:
        top_entries = os.listdir(path)
        logger.info(f"Repo root contents ({len(top_entries)} entries): {sorted(top_entries)[:30]}")
    except Exception as _e:
        logger.warning(f"Could not list repo root: {_e}")

    def should_process_file(file_path: str, use_inclusion: bool, included_dirs: List[str], included_files: List[str],
                           excluded_dirs: List[str], excluded_files: List[str]) -> bool:
        """
        Determine if a file should be processed based on inclusion/exclusion rules.

        Args:
            file_path (str): The file path to check
            use_inclusion (bool): Whether to use inclusion mode
            included_dirs (List[str]): List of directories to include
            included_files (List[str]): List of files to include
            excluded_dirs (List[str]): List of directories to exclude
            excluded_files (List[str]): List of files to exclude

        Returns:
            bool: True if the file should be processed, False otherwise
        """
        file_path_parts = os.path.normpath(file_path).split(os.sep)
        file_name = os.path.basename(file_path)

        if use_inclusion:
            # Inclusion mode: file must be in included directories or match included files
            is_included = False

            # Check if file is in an included directory
            if included_dirs:
                for included in included_dirs:
                    clean_included = included.strip("./").rstrip("/")
                    if clean_included in file_path_parts:
                        is_included = True
                        break

            # Check if file matches included file patterns
            if not is_included and included_files:
                for included_file in included_files:
                    if file_name == included_file or file_name.endswith(included_file):
                        is_included = True
                        break

            # If no inclusion rules are specified for a category, allow all files from that category
            if not included_dirs and not included_files:
                is_included = True
            elif not included_dirs and included_files:
                # Only file patterns specified, allow all directories
                pass  # is_included is already set based on file patterns
            elif included_dirs and not included_files:
                # Only directory patterns specified, allow all files in included directories
                pass  # is_included is already set based on directory patterns

            return is_included
        else:
            # Exclusion mode: file must not be in excluded directories or match excluded files
            is_excluded = False

            # Check if file is in an excluded directory
            for excluded in excluded_dirs:
                clean_excluded = excluded.strip("./").rstrip("/")
                if clean_excluded in file_path_parts:
                    is_excluded = True
                    break

            # Check if file matches excluded file patterns
            if not is_excluded:
                for excluded_file in excluded_files:
                    if file_name == excluded_file:
                        is_excluded = True
                        break

            return not is_excluded

    # Process code files first
    for ext in code_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        kept, skipped_excl, skipped_err = 0, 0, 0
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                skipped_excl += 1
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # Determine if this is an implementation file
                    is_implementation = (
                        not relative_path.startswith("test_")
                        and not relative_path.startswith("app_")
                        and "test" not in relative_path.lower()
                    )

                    # Check token count
                    token_count = count_tokens(content, embedder_type)
                    if token_count > MAX_EMBEDDING_TOKENS * 10:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        skipped_err += 1
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": True,
                            "is_implementation": is_implementation,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
                    kept += 1
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                skipped_err += 1

        if files:
            logger.info(f"  {ext}: found={len(files)}, kept={kept}, skipped_by_filter={skipped_excl}, skipped_by_error={skipped_err}")

    # Then process documentation files
    for ext in doc_extensions:
        files = glob.glob(f"{path}/**/*{ext}", recursive=True)
        for file_path in files:
            # Check if file should be processed based on inclusion/exclusion rules
            if not should_process_file(file_path, use_inclusion_mode, included_dirs, included_files, excluded_dirs, excluded_files):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(file_path, path)

                    # Check token count
                    token_count = count_tokens(content, embedder_type)
                    if token_count > MAX_EMBEDDING_TOKENS:
                        logger.warning(f"Skipping large file {relative_path}: Token count ({token_count}) exceeds limit")
                        continue

                    doc = Document(
                        text=content,
                        meta_data={
                            "file_path": relative_path,
                            "type": ext[1:],
                            "is_code": False,
                            "is_implementation": False,
                            "title": relative_path,
                            "token_count": token_count,
                        },
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    logger.info(f"Found {len(documents)} documents")
    return documents

def prepare_data_pipeline(embedder_type: str = None, is_ollama_embedder: bool = None):
    """
    Creates and returns the data transformation pipeline.

    Args:
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.

    Returns:
        adal.Sequential: The data transformation pipeline
    """
    from api.config import get_embedder_config, get_embedder_type

    # Handle backward compatibility
    if embedder_type is None and is_ollama_embedder is not None:
        embedder_type = 'ollama' if is_ollama_embedder else None
    
    # Determine embedder type if not specified
    if embedder_type is None:
        embedder_type = get_embedder_type()

    splitter = TextSplitter(**configs["text_splitter"])
    embedder_config = get_embedder_config()

    embedder = get_embedder(embedder_type=embedder_type)

    # Choose appropriate processor based on embedder type
    if embedder_type == 'ollama':
        # Use Ollama document processor for single-document processing
        embedder_transformer = OllamaDocumentProcessor(embedder=embedder)
    else:
        # Use batch processing for OpenAI and Google embedders
        batch_size = embedder_config.get("batch_size", 500)
        embedder_transformer = ToEmbeddings(
            embedder=embedder, batch_size=batch_size
        )

    data_transformer = adal.Sequential(
        splitter, embedder_transformer
    )  # sequential will chain together splitter and embedder
    return data_transformer

def transform_documents_and_save_to_db(
    documents: List[Document], db_path: str, embedder_type: str = None, is_ollama_embedder: bool = None
) -> LocalDB:
    """
    Transforms a list of documents and saves them to a local database.

    Args:
        documents (list): A list of `Document` objects.
        db_path (str): The path to the local database file.
        embedder_type (str, optional): The embedder type ('openai', 'google', 'ollama').
                                     If None, will be determined from configuration.
        is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                           If None, will be determined from configuration.
    """
    # Get the data transformer
    data_transformer = prepare_data_pipeline(embedder_type, is_ollama_embedder)

    # Save the documents to a local database
    db = LocalDB()
    db.register_transformer(transformer=data_transformer, key="split_and_embed")
    db.load(documents)
    db.transform(key="split_and_embed")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_state(filepath=db_path)
    return db

def get_github_file_content(repo_url: str, file_path: str, access_token: str = None, branch: str = None) -> str:
    """
    Retrieves the content of a file from a GitHub repository using the GitHub API.
    Supports both public GitHub (github.com) and GitHub Enterprise (custom domains).
    
    Args:
        repo_url (str): The URL of the GitHub repository 
                       (e.g., "https://github.com/username/repo" or "https://github.company.com/username/repo")
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): GitHub personal access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not a valid GitHub URL
    """
    try:
        # Parse the repository URL to support both github.com and enterprise GitHub
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitHub repository URL")

        # Check if it's a GitHub-like URL structure
        path_parts = parsed_url.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format - expected format: https://domain/owner/repo")

        owner = path_parts[-2]
        repo = path_parts[-1].replace(".git", "")

        # Determine the API base URL
        if parsed_url.netloc == "github.com":
            # Public GitHub
            api_base = "https://api.github.com"
        else:
            # GitHub Enterprise - API is typically at https://domain/api/v3/
            api_base = f"{parsed_url.scheme}://{parsed_url.netloc}/api/v3"
        
        # Use GitHub API to get file content
        # The API endpoint for getting file content is: /repos/{owner}/{repo}/contents/{path}
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{file_path}"
        if branch:
            api_url += f"?ref={branch}"

        # Fetch file content from GitHub API
        headers = {}
        if access_token:
            headers["Authorization"] = f"token {access_token}"
        logger.info(f"Fetching file content from GitHub API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")
        try:
            content_data = response.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response from GitHub API")

        # Check if we got an error response
        if "message" in content_data and "documentation_url" in content_data:
            raise ValueError(f"GitHub API error: {content_data['message']}")

        # GitHub API returns file content as base64 encoded string
        if "content" in content_data and "encoding" in content_data:
            if content_data["encoding"] == "base64":
                # The content might be split into lines, so join them first
                content_base64 = content_data["content"].replace("\n", "")
                content = base64.b64decode(content_base64).decode("utf-8")
                return content
            else:
                raise ValueError(f"Unexpected encoding: {content_data['encoding']}")
        else:
            raise ValueError("File content not found in GitHub API response")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_gitlab_file_content(repo_url: str, file_path: str, access_token: str = None, branch: str = None) -> str:
    """
    Retrieves the content of a file from a GitLab repository (cloud or self-hosted).

    Args:
        repo_url (str): The GitLab repo URL (e.g., "https://gitlab.com/username/repo" or "http://localhost/group/project")
        file_path (str): File path within the repository (e.g., "src/main.py")
        access_token (str, optional): GitLab personal access token

    Returns:
        str: File content

    Raises:
        ValueError: If anything fails
    """
    try:
        # Parse and validate the URL
        parsed_url = urlparse(repo_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Not a valid GitLab repository URL")

        gitlab_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        if parsed_url.port not in (None, 80, 443):
            gitlab_domain += f":{parsed_url.port}"
        path_parts = parsed_url.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("Invalid GitLab URL format — expected something like https://gitlab.domain.com/group/project")

        # Build project path and encode for API
        project_path = "/".join(path_parts).replace(".git", "")
        encoded_project_path = quote(project_path, safe='')

        # Encode file path
        encoded_file_path = quote(file_path, safe='')

        # Try to get the default branch from the project info
        if branch:
            default_branch = branch
            logger.info(f"Using specified branch: {default_branch}")
        else:
            default_branch = None
            try:
                project_info_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}"
                project_headers = {}
                if access_token:
                    project_headers["PRIVATE-TOKEN"] = access_token
                
                project_response = requests.get(project_info_url, headers=project_headers)
                if project_response.status_code == 200:
                    project_data = project_response.json()
                    default_branch = project_data.get('default_branch', 'main')
                    logger.info(f"Found default branch: {default_branch}")
                else:
                    logger.warning(f"Could not fetch project info, using 'main' as default branch")
                    default_branch = 'main'
            except Exception as e:
                logger.warning(f"Error fetching project info: {e}, using 'main' as default branch")
                default_branch = 'main'

        api_url = f"{gitlab_domain}/api/v4/projects/{encoded_project_path}/repository/files/{encoded_file_path}/raw?ref={default_branch}"
        # Fetch file content from GitLab API
        headers = {}
        if access_token:
            headers["PRIVATE-TOKEN"] = access_token
        logger.info(f"Fetching file content from GitLab API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            content = response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

        # Check for GitLab error response (JSON instead of raw file)
        if content.startswith("{") and '"message":' in content:
            try:
                error_data = json.loads(content)
                if "message" in error_data:
                    raise ValueError(f"GitLab API error: {error_data['message']}")
            except json.JSONDecodeError:
                pass

        return content

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")

def get_bitbucket_file_content(repo_url: str, file_path: str, access_token: str = None, branch: str = None) -> str:
    """
    Retrieves the content of a file from a Bitbucket repository.

    Supports both Bitbucket Cloud (bitbucket.org) and Bitbucket Server / Data Center
    (self-hosted instances). PAT (Personal Access Token) authentication is supported
    for both variants.

    Args:
        repo_url (str): The URL of the Bitbucket repository.
                        Cloud:  "https://bitbucket.org/username/repo"
                        Server: "https://bitbucket.example.com/projects/PROJ/repos/myrepo"
                                or the SCM clone URL  "https://bitbucket.example.com/scm/PROJ/myrepo"
        file_path (str): The path to the file within the repository (e.g., "src/main.py")
        access_token (str, optional): PAT for private repositories

    Returns:
        str: The content of the file as a string
    """
    try:
        parsed = urlparse(repo_url.rstrip('/'))
        hostname = parsed.hostname or ''
        path_parts = [p for p in parsed.path.split('/') if p]

        is_cloud = hostname in ('bitbucket.org', 'www.bitbucket.org')

        if is_cloud:
            # ── Bitbucket Cloud ──────────────────────────────────────────────────
            if len(path_parts) < 2:
                raise ValueError("Invalid Bitbucket Cloud URL format")

            owner = path_parts[-2]
            repo = path_parts[-1].replace(".git", "")

            # Resolve default branch via Cloud REST API
            if branch:
                default_branch = branch
                logger.info(f"Using specified branch: {default_branch}")
            else:
                default_branch = 'main'
                try:
                    repo_info_url = f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
                    repo_headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
                    repo_response = requests.get(repo_info_url, headers=repo_headers, timeout=10)
                    if repo_response.status_code == 200:
                        default_branch = repo_response.json().get('mainbranch', {}).get('name', 'main')
                        logger.info(f"Found default branch: {default_branch}")
                    else:
                        logger.warning("Could not fetch Bitbucket Cloud repo info, using 'main'")
                except Exception as e:
                    logger.warning(f"Error fetching Bitbucket Cloud repo info: {e}, using 'main'")

            api_url = (
                f"https://api.bitbucket.org/2.0/repositories/{owner}/{repo}"
                f"/src/{default_branch}/{file_path}"
            )
            headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

        else:
            # ── Bitbucket Server / Data Center (self-hosted) ─────────────────────
            base_url = f"{parsed.scheme}://{parsed.netloc}"

            # Parse project key and repo slug from the URL.
            # Supported formats:
            #   Browse URL : /projects/{projectKey}/repos/{repoSlug}[/browse]
            #   Clone URL  : /scm/{projectKey}/{repoSlug}[.git]
            project_key = None
            repo_slug = None

            if 'repos' in path_parts:
                repos_idx = path_parts.index('repos')
                if repos_idx + 1 < len(path_parts):
                    repo_slug = path_parts[repos_idx + 1].replace('.git', '')
                    project_key = path_parts[repos_idx - 1] if repos_idx > 0 else None
            elif 'scm' in path_parts:
                scm_idx = path_parts.index('scm')
                if scm_idx + 2 < len(path_parts):
                    project_key = path_parts[scm_idx + 1]
                    repo_slug = path_parts[scm_idx + 2].replace('.git', '')

            if not project_key or not repo_slug:
                # Generic fallback: treat last two path segments as project/repo
                if len(path_parts) >= 2:
                    project_key = path_parts[-2]
                    repo_slug = path_parts[-1].replace('.git', '')
                else:
                    raise ValueError(
                        "Could not parse project key and repo slug from Bitbucket Server URL. "
                        "Expected format: /projects/{PROJECT}/repos/{REPO} or /scm/{PROJECT}/{REPO}"
                    )

            # Resolve default branch via Bitbucket Server REST API 1.0
            if branch:
                default_branch = branch
                logger.info(f"Using specified branch: {default_branch}")
            else:
                default_branch = 'main'
                try:
                    branch_api_url = (
                        f"{base_url}/rest/api/1.0/projects/{project_key}"
                        f"/repos/{repo_slug}/branches/default"
                    )
                    info_headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
                    branch_response = requests.get(branch_api_url, headers=info_headers, timeout=10)
                    if branch_response.status_code == 200:
                        default_branch = branch_response.json().get('displayId', 'main')
                        logger.info(f"Found default branch: {default_branch}")
                    else:
                        logger.warning(
                            f"Could not fetch default branch from Bitbucket Server "
                            f"(HTTP {branch_response.status_code}), using 'main'"
                        )
                except Exception as e:
                    logger.warning(f"Error fetching Bitbucket Server branch info: {e}, using 'main'")

            # Raw file content endpoint for Bitbucket Server REST API 1.0
            api_url = (
                f"{base_url}/rest/api/1.0/projects/{project_key}"
                f"/repos/{repo_slug}/raw/{file_path}?at={default_branch}"
            )
            headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

        # ── Fetch file content ───────────────────────────────────────────────────
        logger.info(f"Fetching file content from Bitbucket API: {api_url}")
        try:
            response = requests.get(api_url, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 404:
                raise ValueError("File not found on Bitbucket. Please check the file path and repository.")
            elif response.status_code == 401:
                raise ValueError("Unauthorized access to Bitbucket. Please check your access token.")
            elif response.status_code == 403:
                raise ValueError("Forbidden access to Bitbucket. You might not have permission to access this file.")
            elif response.status_code == 500:
                raise ValueError("Internal server error on Bitbucket. Please try again later.")
            else:
                response.raise_for_status()
                return response.text
        except RequestException as e:
            raise ValueError(f"Error fetching file content: {e}")

    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def get_file_content(repo_url: str, file_path: str, repo_type: str = None, access_token: str = None, branch: str = None) -> str:
    """
    Retrieves the content of a file from a Git repository (GitHub or GitLab).

    Args:
        repo_type (str): Type of repository
        repo_url (str): The URL of the repository
        file_path (str): The path to the file within the repository
        access_token (str, optional): Access token for private repositories

    Returns:
        str: The content of the file as a string

    Raises:
        ValueError: If the file cannot be fetched or if the URL is not valid
    """
    if repo_type == "github":
        return get_github_file_content(repo_url, file_path, access_token, branch)
    elif repo_type == "gitlab":
        return get_gitlab_file_content(repo_url, file_path, access_token, branch)
    elif repo_type == "bitbucket":
        return get_bitbucket_file_content(repo_url, file_path, access_token, branch)
    else:
        raise ValueError("Unsupported repository type. Only GitHub, GitLab, and Bitbucket are supported.")

class DatabaseManager:
    """
    Manages the creation, loading, transformation, and persistence of LocalDB instances.
    """

    def __init__(self):
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def prepare_database(self, repo_url_or_path: str, repo_type: Optional[str] = None, access_token: Optional[str] = None,
                         embedder_type: str = None, is_ollama_embedder: bool = None,
                         excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                         included_dirs: List[str] = None, included_files: List[str] = None,
                         branch: Optional[str] = None) -> List[Document]:
        """
        Create a new database from the repository.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
            embedder_type (str, optional): Embedder type to use ('openai', 'google', 'ollama').
                                         If None, will be determined from configuration.
            is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                               If None, will be determined from configuration.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Document]: List of Document objects
        """
        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        
        self.reset_database()
        self._create_repo(repo_url_or_path, repo_type, access_token, branch)
        return self.prepare_db_index(embedder_type=embedder_type, excluded_dirs=excluded_dirs, excluded_files=excluded_files,
                                   included_dirs=included_dirs, included_files=included_files)

    def reset_database(self):
        """
        Reset the database to its initial state.
        """
        self.db = None
        self.repo_url_or_path = None
        self.repo_paths = None

    def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: Optional[str], branch: Optional[str] = None) -> str:
        # Extract owner/project and repo name to create a unique local identifier.
        url_parts = repo_url_or_path.rstrip('/').split('/')

        if repo_type == "bitbucket":
            # Bitbucket Cloud  : https://bitbucket.org/{owner}/{repo}
            # Bitbucket Server : https://{host}/projects/{PROJECT}/repos/{repo}
            #                    https://{host}/scm/{PROJECT}/{repo}[.git]
            parsed = urlparse(repo_url_or_path.rstrip('/'))
            path_parts = [p for p in parsed.path.split('/') if p]
            is_cloud = (parsed.hostname or '') in ('bitbucket.org', 'www.bitbucket.org')

            if is_cloud and len(path_parts) >= 2:
                owner = path_parts[-2]
                repo = path_parts[-1].replace('.git', '')
                repo_name = f"{owner}_{repo}"
            elif 'repos' in path_parts:
                # Server browse URL: /projects/{PROJECT}/repos/{repo}
                repos_idx = path_parts.index('repos')
                project = path_parts[repos_idx - 1] if repos_idx > 0 else 'unknown'
                repo = path_parts[repos_idx + 1].replace('.git', '') if repos_idx + 1 < len(path_parts) else 'unknown'
                repo_name = f"{project}_{repo}"
            elif 'scm' in path_parts:
                # Server clone URL: /scm/{PROJECT}/{repo}[.git]
                scm_idx = path_parts.index('scm')
                project = path_parts[scm_idx + 1] if scm_idx + 1 < len(path_parts) else 'unknown'
                repo = path_parts[scm_idx + 2].replace('.git', '') if scm_idx + 2 < len(path_parts) else 'unknown'
                repo_name = f"{project}_{repo}"
            elif len(path_parts) >= 2:
                owner = path_parts[-2]
                repo = path_parts[-1].replace('.git', '')
                repo_name = f"{owner}_{repo}"
            else:
                repo_name = url_parts[-1].replace('.git', '')
        elif repo_type in ["github", "gitlab"] and len(url_parts) >= 5:
            # GitHub URL format: https://github.com/owner/repo
            # GitLab URL format: https://gitlab.com/owner/repo or https://gitlab.com/group/subgroup/repo
            owner = url_parts[-2]
            repo = url_parts[-1].replace(".git", "")
            repo_name = f"{owner}_{repo}"
        else:
            repo_name = url_parts[-1].replace(".git", "")
        # Append branch suffix so different branches get separate cache dirs
        if branch:
            safe_branch = branch.replace('/', '_').replace('\\', '_')
            repo_name = f"{repo_name}__{safe_branch}"
        return repo_name

    def _create_repo(self, repo_url_or_path: str, repo_type: Optional[str] = None, access_token: Optional[str] = None, branch: Optional[str] = None) -> None:
        """
        Download and prepare all paths.
        Paths:
        ~/.adalflow/repos/{owner}_{repo_name} (for url, local path will be the same)
        ~/.adalflow/databases/{owner}_{repo_name}.pkl

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories
        """
        logger.info(f"Preparing repo storage for {repo_url_or_path}...")

        # Use a stable cache key suffix when no branch is specified.
        # We use "__default" so that all callers that omit the branch
        # will hit the same cache entry.  The actual git clone will
        # fetch the remote's default branch (whatever that is).
        if not branch:
            branch = None  # keep None for git clone (auto-detect)
            cache_branch = "default"
            logger.info("No branch specified, will clone remote default branch (cache key suffix: 'default')")
        else:
            cache_branch = branch

        try:
            # Strip whitespace to handle URLs with leading/trailing spaces
            repo_url_or_path = repo_url_or_path.strip()
            
            root_path = get_adalflow_default_root_path()

            os.makedirs(root_path, exist_ok=True)
            # url
            if repo_url_or_path.startswith("https://") or repo_url_or_path.startswith("http://"):
                # Extract the repository name from the URL (branch included for unique cache key)
                repo_name = self._extract_repo_name_from_url(repo_url_or_path, repo_type, cache_branch)
                logger.info(f"Extracted repo name: {repo_name}")

                save_repo_dir = os.path.join(root_path, "repos", repo_name)

                # Check if the repository directory already exists with actual content
                if not (os.path.exists(save_repo_dir) and _repo_has_content(save_repo_dir)):
                    # Remove broken/empty clone if it exists (e.g. only .git dir)
                    if os.path.exists(save_repo_dir) and os.listdir(save_repo_dir):
                        logger.warning(f"Repository at {save_repo_dir} has no content files. Removing stale clone.")
                        import shutil
                        shutil.rmtree(save_repo_dir)
                    download_repo(repo_url_or_path, save_repo_dir, repo_type, access_token, branch)
                else:
                    logger.info(f"Repository already exists at {save_repo_dir}. Using existing repository.")
            else:  # local path
                repo_name = os.path.basename(repo_url_or_path)
                save_repo_dir = repo_url_or_path

            save_db_file = os.path.join(root_path, "databases", f"{repo_name}.pkl")
            os.makedirs(save_repo_dir, exist_ok=True)
            os.makedirs(os.path.dirname(save_db_file), exist_ok=True)

            self.repo_paths = {
                "save_repo_dir": save_repo_dir,
                "save_db_file": save_db_file,
            }
            self.repo_url_or_path = repo_url_or_path
            logger.info(f"Repo paths: {self.repo_paths}")

        except Exception as e:
            logger.error(f"Failed to create repository structure: {e}")
            raise

    def prepare_db_index(self, embedder_type: str = None, is_ollama_embedder: bool = None, 
                        excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                        included_dirs: List[str] = None, included_files: List[str] = None) -> List[Document]:
        """
        Prepare the indexed database for the repository.

        Args:
            embedder_type (str, optional): Embedder type to use ('openai', 'google', 'ollama').
                                         If None, will be determined from configuration.
            is_ollama_embedder (bool, optional): DEPRECATED. Use embedder_type instead.
                                               If None, will be determined from configuration.
            excluded_dirs (List[str], optional): List of directories to exclude from processing
            excluded_files (List[str], optional): List of file patterns to exclude from processing
            included_dirs (List[str], optional): List of directories to include exclusively
            included_files (List[str], optional): List of file patterns to include exclusively

        Returns:
            List[Document]: List of Document objects
        """
        def _embedding_vector_length(doc: Document) -> int:
            vector = getattr(doc, "vector", None)
            if vector is None:
                return 0
            try:
                if hasattr(vector, "shape"):
                    if len(vector.shape) == 0:
                        return 0
                    return int(vector.shape[-1])
                if hasattr(vector, "__len__"):
                    return int(len(vector))
            except Exception:
                return 0
            return 0

        # Handle backward compatibility
        if embedder_type is None and is_ollama_embedder is not None:
            embedder_type = 'ollama' if is_ollama_embedder else None
        # check the database
        if self.repo_paths and os.path.exists(self.repo_paths["save_db_file"]):
            logger.info("Loading existing database...")
            try:
                self.db = LocalDB.load_state(self.repo_paths["save_db_file"])
                documents = self.db.get_transformed_data(key="split_and_embed")
                if documents:
                    lengths = [_embedding_vector_length(doc) for doc in documents]
                    non_empty = sum(1 for n in lengths if n > 0)
                    empty = len(lengths) - non_empty
                    sample_sizes = sorted({n for n in lengths if n > 0})[:3]
                    logger.info(
                        "Loaded %s documents from existing database (embeddings: %s non-empty, %s empty; sample_dims=%s)",
                        len(documents),
                        non_empty,
                        empty,
                        sample_sizes,
                    )

                    if non_empty == 0:
                        logger.warning(
                            "Existing database contains no usable embeddings. Rebuilding embeddings..."
                        )
                    else:
                        return documents
            except Exception as e:
                logger.error(f"Error loading existing database: {e}")
                # Continue to create a new database

        # prepare the database
        logger.info("Creating new database...")
        documents = read_all_documents(
            self.repo_paths["save_repo_dir"],
            embedder_type=embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        self.db = transform_documents_and_save_to_db(
            documents, self.repo_paths["save_db_file"], embedder_type=embedder_type
        )
        logger.info(f"Total documents: {len(documents)}")
        transformed_docs = self.db.get_transformed_data(key="split_and_embed")
        logger.info(f"Total transformed documents: {len(transformed_docs)}")
        return transformed_docs

    def prepare_retriever(self, repo_url_or_path: str, repo_type: str = None, access_token: str = None):
        """
        Prepare the retriever for a repository.
        This is a compatibility method for the isolated API.

        Args:
            repo_type(str): Type of repository
            repo_url_or_path (str): The URL or local path of the repository
            access_token (str, optional): Access token for private repositories

        Returns:
            List[Document]: List of Document objects
        """
        return self.prepare_database(repo_url_or_path, repo_type, access_token)
