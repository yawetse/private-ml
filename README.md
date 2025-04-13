# private-ml
privacy preserving machine learning


To install

```bash
#recommended to use a venv
python3 -m venv .venv
source .venv/bin/activate
#end of recommended
pip install -r requirements.txt
```

Before checking into GitHub, please update requirements.txt via:

```bash
pip freeze > requirements.txt
```

## install Git Large File Storage

- Run the following command in your repository to initialize Git LFS:
  
```bash
git lfs install
# Specify which files you want to track with Git LFS. For example, to track all `.psd` files, run:
git lfs track "*.csv"
# This command adds a pattern to the `.gitattributes` file in your repository.
# Add the files you want to store in Git LFS to your repository:
git add <file>
git commit -m "Add large files to Git LFS"
git push origin <branch>
```
