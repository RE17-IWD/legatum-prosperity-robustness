# How to put this repository on GitHub

The repository is fully built and already has a clean first commit. You only need to create an empty repo on GitHub and push. Two ways, pick one.

Your username below is shown as ADRIAN; replace it with your actual GitHub username. Suggested repo name: `legatum-prosperity-robustness`.

---

## Option A: GitHub website + command line (most reliable)

1. Unzip the downloaded `legatum-prosperity-robustness.zip`. You get a folder with a hidden `.git` directory already inside it, so the commit history is preserved.

2. On github.com, click New repository. Name it `legatum-prosperity-robustness`. Set it to Public. Do NOT check "Add a README," "Add .gitignore," or "Add a license" (the repo already has them). Click Create repository.

3. In a terminal, go into the unzipped folder and run:

       cd legatum-prosperity-robustness
       git remote add origin https://github.com/ADRIAN/legatum-prosperity-robustness.git
       git branch -M main
       git push -u origin main

   If prompted for a password, GitHub no longer accepts your account password on the command line. Create a Personal Access Token (github.com, Settings, Developer settings, Personal access tokens, Tokens (classic), Generate new token, give it `repo` scope) and paste the token as the password.

4. Refresh the GitHub page. All files should be there, and `data/` will contain only its README, which is correct.

---

## Option B: GitHub Desktop (no command line)

1. Install GitHub Desktop (desktop.github.com) and sign in.
2. Unzip the folder.
3. In GitHub Desktop: File, Add Local Repository, choose the unzipped folder. It will detect the existing git history.
4. Click Publish repository. Uncheck "Keep this code private" so it is public. Publish.

---

## After pushing

1. Confirm the Legatum `.xlsx` is NOT on GitHub. Only `data/README.md` should appear in the `data` folder. The `.gitignore` already prevents the data file from being uploaded.

2. Copy your repo URL, which will be `https://github.com/ADRIAN/legatum-prosperity-robustness`.

3. Optional but recommended, mint a DOI:
   - Go to zenodo.org, log in with GitHub.
   - Settings, GitHub, find the repo, flip the switch to On.
   - Back on GitHub, create a release: Releases, Create a new release, tag `v1.0`, title `v1.0`, Publish release.
   - Zenodo automatically archives it and gives a DOI badge. Copy the DOI.

4. Paste the GitHub URL and the Zenodo DOI into the paper:
   - Open `main.tex`, find `[GITHUB URL]` and `[ZENODO DOI]` in the Data and Code Availability section, replace them, recompile.
   - Do the same in the cover letter where the links appear.
   - If you skip Zenodo, just use the GitHub URL and remove the Zenodo clause.

---

## What is in the repository

    README.md                  overview, how to run, method summary
    LICENSE                    MIT (code only; data excluded)
    requirements.txt           pinned Python dependencies
    .gitignore                 excludes the Legatum data and scratch files
    data/README.md             how to obtain the dataset (data itself not included)
    src/analysis.py            baseline replication, Monte Carlo, Sobol'
    src/figures.py             regenerates all five figures + importance table
    figures/                   the five figures (PNG)
    outputs/                   precomputed result CSVs and summary.json

The code was tested end to end from a clean copy: with the dataset placed in `data/`, running `python src/analysis.py` then `python src/figures.py` reproduces every result and figure, with a fixed seed. The headline numbers match the paper (median 90% rank range 22.0; average rank shift 5.73 under flat Dirichlet).
