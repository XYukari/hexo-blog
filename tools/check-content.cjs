'use strict';

const Hexo = require('hexo');

(async () => {
  const hexo = new Hexo(process.cwd(), { silent: true });
  try {
    await hexo.init();
    await hexo.load();
    const posts = hexo.locals.get('posts');
    hexo.contentPolicy.validate(posts);
    console.log(`Validated ${posts.length} posts against the shared language and taxonomy vocabulary.`);
  } finally {
    await hexo.exit();
  }
})().catch(error => {
  console.error(error.message);
  process.exitCode = 1;
});
