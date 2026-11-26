# Uploading the project to GitHub

Upload the extracted project files so that `README.md` appears at the repository root. Keep the full submission ZIP separately as the data-and-model bundle.

## Prepare the repository

1. Create an empty repository in the intended GitHub account, or use the existing project repository.
2. Place the extracted project files in its local working folder. Keep the supplied `.gitignore` file.
3. Review the files selected for upload. Code, CSV/JSON result tables, documentation, the executed notebook, figures, report and poster belong in the repository. Large expression arrays, encoder weights and trained checkpoints remain local.
4. Connect the local repository to the URL shown by GitHub for that repository. No remote URL is configured in this package.

Before committing, confirm that the ignore rules recognize representative data and model files:

```sh
git check-ignore MLP_PRnet_controls/prepared/targets.npy extension/balanced/weights/Residual_42.pt
git status --short
```

The first command should print both file paths. Review the second command's output before staging files; model weights and expression arrays should not appear as new files for upload.

## Keep the history clear

Suggested commits for an initial upload are:

- `Add benchmark code and data documentation`
- `Add evaluation results and experiment settings`
- `Add analysis notebook, report and poster`

Commit only the files belonging to each step. Use actual commit dates and the account of the contributor making the commit. Preserve existing history when updating a repository. The initial upload records the completed project; later commits can describe specific changes to methods or results.

## Share the data and models separately

GitHub blocks files larger than 100 MiB in regular Git repositories. The supplied ignore rules exclude the binary experiment assets from Git while keeping them in the submission ZIP. See [GitHub's large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Store the full bundle in an appropriate data repository or shared storage location. Once that location exists, add its actual download link to [DATA.md](DATA.md). A fresh clone can display the executed notebook and regenerate figures and reports from the versioned result tables. Prediction, full verification, evaluation and training require the additional files described there.

After uploading, open the repository on GitHub and check the README links, pipeline image, notebook, report and poster. Retain the PRnet license, font license and published-figure attribution with their files.
