# Repository Guidelines

## Project Structure

This is a Hexo 8 blog using Fluid. Articles live in `source/_posts/`, with adjacent folders for article-specific images. Shared images and the personal avatar belong in `source/images/`. `scaffolds/` contains writing templates.

Keep site settings in `_config.yml` and personal theme overrides in `_config.fluid.yml`. `themes/fluid/` contains upstream theme files plus the documented document-language fix in `layout/layout.ejs`. Avoid adding personal settings there.

`source/_data/taxonomy.json` is the shared vocabulary and archive-slug registry. `scripts/` enforces content policy and generates legacy archive redirects; `tools/` contains local checks and lossless image optimization.

Generated `public/`, `db.json`, `node_modules/`, `.deploy_git/`, and `.local-backups/` are ignored. Never edit generated pages or commit backups and deployment credentials.

## Development Commands

Use Node.js 24 LTS with npm. Run from the repository root:

- `npm ci`: Install locked dependencies.
- `npx hexo new post "article-slug"`: Create an article and its asset folder.
- `npm run server`: Preview locally.
- `npm run check`: Validate article languages, categories, and tags.
- `npm run clean && npm run build`: Rebuild the static site.
- `npm run optimize:images`: Recompress PNG data losslessly, preserving dimensions and metadata.
- `npm run deploy`: Publish generated pages via HTTPS, only when explicitly requested. Build first.

## Writing, Languages, and Style

Use UTF-8, two-space indentation, and descriptive filenames. Follow Fluid's EditorConfig and ESLint conventions for theme code: single-quoted JavaScript strings and semicolons.

The interface stays English. Set each article's `lang` to `en`, `zh-CN`, or `ja`; existing Chinese titles and prose remain Chinese. New scaffolds default to English. Select categories and tags from the vocabulary; extend that registry intentionally instead of creating near-duplicate labels. Select a category before building a new post.

Preserve published filenames, dates, and permalinks. Keep legacy redirects when changing taxonomy slugs. Use `math: true` for formulas; the theme loads MathJax centrally. Reference article images with `{% asset_img diagram.png %}` and provide meaningful descriptions in the article's language.

## Validation and Contributions

Run content checks and a clean build. Preview navigation, article images, formulas, and mobile layout; language changes must preserve English UI and correct article-language metadata. No coverage threshold is configured.

Use short, focused commit subjects, following history. PRs should describe changes, list validation, link relevant issues, and include screenshots for visual changes. Keep unrelated edits out of commits.
