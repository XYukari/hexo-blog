# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Hexo 8 blog using the Fluid theme.

- `source/_posts/`: Markdown articles; each article’s images belong in an adjacent folder with the same basename.
- `source/about/index.md`: About page.
- `scaffolds/`: Templates for new posts, pages, and drafts.
- `themes/fluid/`: EJS layouts, JavaScript helpers, Stylus styles, translations, and theme assets.
- `_config.yml`: Site settings and deployment configuration; `themes/fluid/_config.yml` controls theme features.
- `public/`, `db.json`, and `.deploy_git/`: Generated output, cache, and deployment checkout; do not edit or commit them.

## Build, Test, and Development Commands

Use Node.js 24 LTS with npm; a global Hexo installation is unnecessary. Run commands from the repository root:

- `npm ci`: Install dependencies from `package-lock.json`.
- `npm run server`: Start the local preview server.
- `npm run build`: Generate the static site into `public/`.
- `npm run clean`: Clear generated files and Hexo’s cache before rebuilding when output appears stale.
- `npx hexo new post "Article-Title"`: Create an article using the post scaffold.
- `npm run deploy`: Publish through the configured Git deployer. Run only when publication is explicitly requested.

## Coding Style & Naming Conventions

Use two-space indentation for YAML and theme code. Follow Fluid’s `.editorconfig` and `.eslintrc`: UTF-8, LF endings, single-quoted JavaScript strings, and semicolons. No root formatter or lint command is configured.

Keep descriptive article filenames consistent with existing Chinese titles or hyphenated problem names, such as `CF1878D-Reverse-Madness.md`. Preserve YAML front matter with `title`, `date`, and relevant `category`/`tags`. Use fenced code blocks with language identifiers and keep image paths aligned with the article’s asset folder.

## Testing Guidelines

No automated test suite or coverage threshold is configured. Run `npm run build` after changes and inspect affected pages with `npm run server`. Check images, links, code highlighting, math rendering, navigation, and mobile layout. For theme changes, inspect both an article and the homepage.

## Commit & Pull Request Guidelines

History uses short, topic-focused subjects such as `add ViT&Swin` and `leetcode 3721`; no strict prefix convention is established. Keep commits focused and stage specific files to preserve unrelated work.

Pull requests should describe the affected posts or theme behavior, list validation performed, link relevant issues, and include screenshots for visual changes. Keep deployment keys and credentials out of commits.
