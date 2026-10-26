/* global hexo */
'use strict';

const fs = require('fs');
const path = require('path');
const taxonomy = JSON.parse(fs.readFileSync(path.join(hexo.base_dir, 'source/_data/taxonomy.json'), 'utf8'));

// One vocabulary supplies both validation and stable, unambiguous archive URLs.
for (const [kind, option] of [['categories', 'category_map'], ['tags', 'tag_map']]) {
  const slugs = new Set();
  for (const [name, entry] of Object.entries(taxonomy[kind])) {
    if (!/^[\x20-\x7e]+$/.test(name) || !/^[A-Za-z0-9-]+$/.test(entry.slug)
      || slugs.has(entry.slug.toLowerCase())) {
      throw new Error(`Invalid or duplicate ${kind} vocabulary entry: ${name}`);
    }
    slugs.add(entry.slug.toLowerCase());
  }
  hexo.config[option] = Object.fromEntries(Object.entries(taxonomy[kind]).map(([name, entry]) => [name, entry.slug]));
}

function validate(posts) {
  const errors = [];
  posts.forEach(post => {
    if (!taxonomy.languages.includes(post.lang)) {
      errors.push(`${post.source}: set lang to en, zh-CN, or ja`);
    }
    for (const kind of ['categories', 'tags']) {
      post[kind].forEach(item => {
        if (!Object.hasOwn(taxonomy[kind], item.name)) {
          errors.push(`${post.source}: unknown ${kind} value "${item.name}"; use source/_data/taxonomy.json`);
        }
      });
    }
  });
  if (errors.length) throw new Error(errors.join('\n'));
}

hexo.contentPolicy = { taxonomy, validate };
hexo.extend.filter.register('before_generate', function() {
  validate(this.locals.get('posts'));
});

// Article language controls HTML/metadata; the site interface stays English.
hexo.extend.filter.register('template_locals', function(locals) {
  locals.__ = this.theme.i18n.__('en');
  locals._p = this.theme.i18n._p('en');
  return locals;
}, 100);

hexo.extend.generator.register('legacy_taxonomy_redirects', function() {
  return taxonomy.legacy_routes.map(entry => {
    const slug = taxonomy[entry.kind][entry.name].slug;
    const target = new URL(`${entry.kind}/${slug}/`, this.config.url + '/').href;
    return {
      path: entry.path,
      data: '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        + `<meta http-equiv="refresh" content="0;url=${target}">`
        + `<link rel="canonical" href="${target}"><title>Moved</title></head>`
        + `<body><a href="${target}">This archive has moved.</a></body></html>`
    };
  });
});

// The theme uses this file as a template; only the rendered root index is public.
hexo.extend.filter.register('after_generate', function() {
  this.route.remove('xml/local-search.xml');
});
