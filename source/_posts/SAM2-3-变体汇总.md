---
title: SAM2/3 变体汇总
date: 2026-05-16 14:57:02
categories:
    - 科研
tags:
    - 多模态
    - LLM
    - SAM
    - RVOS
    - Ref-AVS
---

## SAMWISE

CVPR2025, RVOS

### Motivation

- 为 SAM2 增加文本理解，从而引入到 RVOS 任务
- SAM2 的 tracker 只能跟踪目标，不能理解跨帧的动作语义
- SAM2 存在 tracking bias，一旦分割错目标很难纠正

### Pipeline

{% asset_img SAMWISE.png %}

### Prompting SAM 2

这一篇是比较早的方法，没有用到 LLM。注入方法是用文本编码器输出 token 投影后拼接到 prompt encoder 输出的 `sparse_embeddings` 上，送入 mask decoder 进行预测，比较平凡。特色在于前处理期间将文本编码器和 SAM2 的图像编码器层间加入了 adapter，通过交叉注意力进行了早期对齐；以及在选取文本 token 注入的时候除了 `[CLS]` 之外还选择了全部动词 token。

#### Adapter

文本编码器是 RoBERTa，共 12 层；图像编码器是 SAM2 的 Hiera 编码器，共 48 层。在上方 pipeline 大图中可以看到，图像编码器每 4 层的输出 $F_k$ 和文本编码器每 1 层的输出 $\varepsilon_k$ 输入 adapter（文中称为 CMT），经过双向注意力等操作后回到各自的路径继续向下。具体代码为：

```python
# models/samwise.py:266-296, _early_fusion_stage()
vis = SAM2.patch_embed(image) + pos_embed
for i, (i_v, i_t) in enumerate(zip(fusion_vis[:-1], fusion_txt[:-1])):
    vis = SAM2_ViT_blocks[i_v_start : i_v_end](vis)
    txt = RoBERTa_layers[i_t_start : i_t_end](txt)
    if i in self.fusion_stages:  # 共 4 个融合阶段
        v, t = CMT_adapter[i](vis, txt)  # HSA + VTA + TVA
        vis = vis + v  # 视觉残差
        txt = txt + t  # 文本残差
```

CMT adapter 是一个瓶颈结构，先下采样，再进行注意力操作，最后上采样，残差连接后输出。下图中把残差连接画到了 Adapter 里，实际代码是在 Adapter 外面计算的，不过只是实现问题。上下采样都是平凡的，主要来看中间的处理。

{% asset_img CMT.png %}

如上图所示，$F_k, \varepsilon_k$ 经过下采样后，分别进入 HSA, VTA, TVA 三个模块：

- HSA：只接收 $F_k$，内部是一个类似 Swin 的多层局部注意力，意在加强时序上的动作建模。
- VTA：以 $F_k$ 为 query，$\varepsilon_k$ 为 key/value 计算交叉注意力 $attn$，$F_k\gets F_k\cdot attn$ 调制 $F_k$。
- TVA：以 $\varepsilon_k$ 为 query，$F_k$ 为 key/value 计算交叉注意力 $attn$，$\varepsilon_k\gets \varepsilon_k\cdot attn$ 调制 $\varepsilon_k$。

VTA 和 TVA 具体代码类似：

```python
attn = multihead_attn(query=tgt, key=other, value=other)
tgt = tgt * attn  # 信息门控，抑制不相关特征
```

#### 注入 SAM 2

图像编码器的输出正常经过 memory attention 进入 mask decoder。文本编码器的输出形为 `[B, L, 768]`，取出 `[CLS]` 和所有动词的 token（作者认为有助于建模动作），经过 MLP 投影之后和 `sparse_embeddings` 进行拼接后送入 mask decoder。

提取 `[CLS]` 的代码如下：

```python
# models/samwise.py:289-292
txt = txt.permute(1, 0, 2)  # [L, B, 768] → [B, L, 768]
state = txt[:, 0]  # CLS token: [B, 768]
if T > 1:
    state = state.repeat_interleave(T, 0)  # 每帧复制
```

提取动词 token 的代码如下，依赖于 spaCy 词性标注 + RoBERTa BPE subtoken 的字符偏移对齐：

```python
# models/samwise.py:229-243, extract_motion_prompts()
docs = [self.nlp_dict(x) for x in captions]   # spaCy 解析每个句子
motion_map = torch.zeros(size=input_ids.shape)  # [B, L] 全零
encoded_input = self.tokenizer(captions, ..., return_offsets_mapping=True)

for caption_index, doc in enumerate(docs):
    roberta_tokens = self.tokenizer.convert_ids_to_tokens(...)
    roberta_offsets = encoded_input['offset_mapping'][caption_index]
    for index, (rt, offset) in enumerate(zip(roberta_tokens, roberta_offsets)):
        start, end = offset
        if rt != '<s>' and rt != '</s>':
            for token in doc:
                # 字符偏移对齐：spaCy token 覆盖 RoBERTa subtoken
                if token.idx <= start and (token.idx + len(token.text)) >= end:
                    if token.pos_ == 'VERB':    # ← 词性判定
                        motion_map[caption_index][index] = 1
```

`[CLS]` 和动词 token 的 MLP 投影层结构相同（3 层，$768 → 256 → 256 → 256$）但是参数独立，代码如下：

```python
# models/sam2/modeling/sam/prompt_encoder.py:69-71
self.project_text = MLP(input_dim=768, hidden_dim=256, output_dim=256, num_layers=3)
self.project_motion_prompts = MLP(input_dim=768, hidden_dim=256, output_dim=256, num_layers=3)

# layers.py:20-32
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    # Linear(768, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 256)
```

拼接到 `sparse_embeddings` 的代码如下：

```python
# models/sam2/modeling/sam/prompt_encoder.py:183-198
sparse_embeddings = torch.empty((bs, 0, 256))
if text is not None:
  text_embeddings = self._embed_text(text)  # [B, 1, 256]
  sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)
if motion is not None:
  motion_embeddings = self._embed_motion(motion)  # [B, N_verbs, 256]
  sparse_embeddings = torch.cat([sparse_embeddings, motion_embeddings], dim=1)
```

最终 `sparse_embeddings` 形状为 `[B, P + 1 + N_verbs, 256]`。注意到这里点的输入不是空的，而是有 2 个可学习的 `not_a_point_embed`，即一个可学习的占位点，所以原本的 `sparse_embeddings` 也不是空的。

### Tracking Bias

作者注意到如果从首帧开始分割，容易在关键帧出现之前分割到错误的目标上（比如指代是“爬树的猫”，但是前几帧两只猫都在睡觉，模型随便选了一只）；由于 SAM2 的 memory，一旦分割错，自己很难纠正。因此，作者对每一帧用 SAM2 图像编码器的原始特征图和 prompt 文本计算交叉注意力，得到一个当前帧 memory-less 的 token $\tau_l$，不包含上下文信息。随后，作者对 memory bank 输出的 token（实际就是 SAM2 原论文中的 object pointers） $\tau_m, \tau_l$ 和一个可学习的 token `[DEC]` 做自注意力，并将 `[DEC]` 通过一个线性分类器。如果分类结果 $p_{detect}>0.5$ 则认定分割错了，应该转换到新目标。

转换的具体方式不是直接放弃错误 mask，换用新 token，因为作者发现这样做缺少上下文信息，语义损失太大，导致后续分割不稳定。因此作者设计了一个 soft alignment，在新目标存在的位置使用新目标，在新目标不存在的位置保留老的跟踪目标。因为这篇做的是流式处理而非离线处理，不能修改前序的分割结果，所以为了保证后续分割结果的准确率也合理。

## OpenWorldSAM

RVOS

### Motivation

- SAM 类分割模型缺乏理解一对多文本指代的能力（同一文本指向多个不重叠的区域，且模型不知道要分割多少个实例）

### Pipeline

{% asset_img OpenWorldSAM.png %}

### Prompting SAM2

OpenWorldSAM 用 BEiT-3 联合编码视觉、文本模态。相比于 CLIP，这个编码器更早融合视觉文本模态，且使用掩码重建学习而非对比学习，更好的学到了图像的结构和细节信息，因此被认为更适合 RVOS/Ref-AVS 任务。BEiT-3 内部将 image patches + text tokens 拼接，通过多路 Transformer。输出序列为 `[CLS] [img_1 ... img_N] [text_1 ... text_L]`，`[:, :1, :]`。此处取到唯一的 `[CLS]` token，其余 token 丢弃。

```python
output = self.mm_extractor.beit3(
    visual_tokens=images_evf,           # [total_prompts, 3, 224, 224]
    textual_tokens=input_ids,           # [total_prompts, L]
    text_padding_position=~attention_masks
)
feat = output["encoder_out"][:, :1, ...]   # [total_prompts, 1, 768]
feat = self.text_hidden_fcs[0](feat)        # [total_prompts, 1, 256]
feat = torch.split(feat, [n_i, ...])        # list of [n_i, 1, 256] per image
```

将 `[CLS]` 经过一个 MLP 层投影到 256 维以适合 SAM2 输入，MLP 代码如下：

```python
text_fc = [
    nn.Linear(in_dim, in_dim),   # 768→768
    nn.ReLU(),
    nn.Linear(in_dim, query_dim) # 768→256
]
text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
# param.requires_grad = True
```

随后将 `[CLS]` 复制 $K=20$ 份，分别加上不同的可学习的位移向量，意为 attend to 符合指代的不同实例，得到 $K$ 个 object queries：

```python
# __init__:
self.positional_tokens = nn.Parameter(torch.randn(20, 256))  # [K, C] 可学习

# forward:
for prompt_idx, prompt_feat in enumerate(img_feat):
    feat_repeated = prompt_feat.expand(20, -1, -1)            # [20, 1, 256]
    feat_with_tokens = feat_repeated + self.positional_tokens.unsqueeze(1)
    # → [20, 1, 256] -- 每份加不同的可学习位置扰动
    batch_feat_with_tokens.append(feat_with_tokens)

batch_feat_with_tokens = torch.cat(batch_feat_with_tokens, dim=0)
# → [n_prompts*20, 256]
```

$K$ 个 object queries 进入 soft-prompting transformer 进行自注意力和对图像特征的交叉注意力：

```python
if self.use_cross_attention:
    img_embed = _features["image_embed"][img_idx]         # [256, 64, 64]
    img_embed = img_embed.flatten(1).transpose(0, 1)      # [4096, 256]
    img_embed = img_embed.unsqueeze(0)                    # [1, 4096, 256]

    original = batch_feat_with_tokens                     # [n*20, 256]
    reshaped = original.unsqueeze(0)                      # [1, n*20, 256]

    enhanced = self.cross_attention_transformer(reshaped, img_embed)
    #   ↑ query: [1, n*20, 256]      ← VLM tokens
    #   ↑ key/value: [1, 4096, 256]  ← SAM2 Hiera features

    batch_feat_with_tokens = original + enhanced.squeeze(0)  # skip connection
```

其中 `cross_attention_transformer` 的单层：

```python
def forward(self, vlm_features, image_embeddings):
    # 1) Self-Attn on VLM tokens
    x = residual + self.self_attn(norm(vlm_features))
    # 2) Cross-Attn: VLM (query) → image (key/value)
    x = residual + self.cross_attn(query=x, key=image_embeddings, value=image_embeddings)
    # 3) MLP
    x = residual + self.mlp(norm(x))
    return x
```

至此得到了 $K$ 个最终的 object queries，与 `sparse_pe` 拼接后送入 mask decoder：

```python
sparse_embeddings, dense_embeddings = self.visual_model.sam_prompt_encoder(
    points=None, boxes=None, masks=None,
    text_embeds=batch_feat_with_tokens,    # [n_prompts*20, 256]
)
```

## MPG-SAM2

ICCV 2025, RVOS

### Motivation

- SAM2 用于 RVOS 时缺乏文本理解能力
- SAM2 缺乏全局上下文（流式模型，离线处理时也看不到后文）

### Pipeline

{% asset_img MPG-SAM2.png %}

### Prompting SAM2

模型使用 BEiT-3 联合编码视频和文本，得到 `[CLS]`、视觉 token、文本 token。`[CLS]` 经过两层 MLP 投影之后拼接 `sparse_pe`，`[CLS]` 和视觉 token 用于生成伪 mask 作为 `dense_pe`，各种 token 还将在后续 SAM2 的 mask decoder 的 fusion 模块中参与调制。

`[CLS]` 用两层 MLP 投影到 SAM2 维度：

```python
# models/evf_sam.py:341
feat = self.text_hidden_fcs[0](feat)  # → [B×T, 1, 256]

text_fc = [
    nn.Linear(in_dim, in_dim),  # 1024 → 1024 (base: 768→768)
    nn.ReLU(),
    nn.Linear(in_dim, out_dim)  # 1024 → 256  (base: 768→256)
]
self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
```

然后与 `sparse_pe` 拼接：

```python
sparse_embeddings, dense_embeddings = self.visual_model.sam_prompt_encoder(
    points=None,
    boxes=None,
    masks=mask_prompt,  # 伪 mask → dense
    text_embeds=feat,  # [CLS] 投影 → sparse
)
# 内部逻辑
if text_embeds is not None:
    sparse_embeddings = torch.cat([sparse_embeddings, text_embeds], dim=1)
```

这里没有使用 dummy points，似乎魔改了 `_get_batch_size` 的逻辑以支持纯 `text_embeds` 的输入。

再来看 `dense_pe` 的生成。首先是对视觉 token 去做自注意力，然后 `mask token`（`[CLS]` 的副本）和视觉 token 做交叉注意力 ：

```python
visual_feat = rearrange(visual_feat, '(b t) hw c -> (t hw) b c', b=batch_size)
mask_token = rearrange(mask_token, '(b t) m c -> (t m) b c', b=batch_size)

for i in range(self.num_vm_layer):    # 1 层
    visual_feat = self.visual_ifi(visual_feat)     # self-attention on visual
    mask_token = self.vmtoken(tgt=mask_token,       # cross-attn: mask_token query
                               memory=visual_feat)  # visual_feat key/value
```

第二步，此时 `mask token` 已经是交叉注意力的结果，变换后与 `visual_feat` 逐元素相乘门控得到 `mask_pred`；`mask_pred` 通过一个线性层后 sigmoid 得到 `dense_pe`。后半部分给 `mask_pred` 加入位置编码，用于后续 HGA 模块注入全局上下文信息。

```python
mask_token = rearrange(mask_token, '(t m) b c -> (b t) m c', t=bt//batch_size)
visual_feat = rearrange(visual_feat, '(t hw) b c -> (b t) hw c', t=bt//batch_size)

mask_token = self.mask_linear(mask_token).unsqueeze(1) # [B×T, 1, 1, 1024]

visual_feat = rearrange(visual_feat, 'bt (h w) c -> bt h w c', h=14, w=14)

mask_pred = visual_feat * mask_token  # 逐元素门控
mask_prompt = torch.sigmoid(self.mask_product(visual_feat * mask_token))  
# [B×T, 14, 14, 1] → rearrange → [B×T, 1, 14, 14] 伪 mask

# 加时间位置编码
num_t = bt // batch_size
pos_inds = torch.arange(num_t, dtype=...)
mp_pos_embed = get_1d_sine_pe(pos_inds, dim=256)  # 1D 正弦 PE

mask_pred = self.meo_linear(mask_pred)  # 1024 → 256
mask_pred = rearrange(mask_pred, '(b t) h w c -> b t c (h w)', t=num_t)
mask_pred = mask_pred + mp_pos_embed  # 加时间 PE
mask_pred = rearrange(mask_pred, 'b t c hw -> b (t hw) c')
```

#### HGA

除了构造 `sparse_pe` 和 `dense_pe`，这篇文章还在 SAM2 的 mask decoder 前后设置了多个 fusion 模块。

##### vtlfusion

在提取完特征后，进入 mask decoder 前，将文本和压缩后的视觉特征拼接成 `vis_text_feat` 作为 key/value，`image_embeddings` 作为 query 计算交叉注意力后逐元素相乘，让视觉特征图吸收文本和全局视觉信息。代码如下：

```python
# mask_decoder.py:148-176
text_feat = rearrange(text_feat, 'b l c -> l b c')
image_embeddings = rearrange(image_embeddings, 'b c h w -> (h w) b c')  # [4096, B, 256]
image_pe = rearrange(image_pe, 'b c h w -> (h w) b c')

# visual_feat 做 patchify：14×14 → 7×7 patches of 2×2
visual_feat = rearrange(visual_feat, 'b t (h w) c -> (b t) c h w', h=14, w=14)
visual_feat = visual_feat.view(bt, c, 7, 2, 7, 2)  # patch_size=2
visual_feat = visual_feat.permute(0,1,3,5,2,4).contiguous()
visual_feat = visual_feat.view(bt, c*4, 7, 7)  # [B×T, 1024, 7, 7]
visual_feat = rearrange(visual_feat, '(b t) c h w -> (t h w) b c', t=t)
visual_feat = self.patch_linear(visual_feat)  # 1024 → 256: [T×49, B, 256]

# 拼接 visual 和 text 作为 memory
vis_text_feat = torch.cat((visual_feat, text_feat), dim=0)  # [T×49+L, B, 256]

# vtlfusion_module: image_embeddings attend to vis_text_feat
image_embeddings = self.vtlfusion_module(
    tgt=image_embeddings,  # query: [4096, B, 256]
    memory=vis_text_feat,  # key/value: [T×49+L, B, 256]
    query_pos=image_pe
)
# 输出: image_embeddings 被 visual+text 特征调制后 reshape 回 [B, 256, 64, 64]
```

##### mmfusion & objfusion

上文中提取了 `mask_pred` 伪 mask，mmfusion 用 `mask_pred` 做 key/value，用 SAM2 自带的 4 个 learnable mask tokens 做 query 计算交叉注意力，用 `mask_pred` 初始化 SAM2 的 mask token；objfusion 则是用 memory bank 中存储的 object pointers 做 key/value，和 mask tokens 做 query 计算交叉注意力，吸收对象级别的特征。代码如下：

```python
# mask_decoder.py:213-273
def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                  dense_prompt_embeddings, mask_pred, ...):
    # SAM2 原始 learnable tokens
    output_tokens = cat([iou_token, mask_tokens])  # [B, 5, 256]

    # 分离 post_tokens (iou_token) 和 mask_init_token (4 mask tokens)
    post_tokens = output_tokens[:, 0:1, :]  # [B, 1, 256]
    mask_init_token = output_tokens[:, 1:5, :]  # [B, 4, 256]

    # mmfusion_module: mask tokens attend to mask_pred
    mask_init_token = rearrange(mask_init_token, 'b m c -> m b c')
    mask_init_token = self.mmfusion_module(
        tgt=mask_init_token,  # query: [4, B, 256]
        memory=mask_pred  # key/value: [T×196, B, 256] (来自 Mask Prior Generator)
    )
    # 如果还有 obj_memory（历史帧信息）
    if obj_memory is not None:
        # objfusion_module: mask tokens attend to obj_memory
        mask_init_token = self.objfusion_module(
            tgt=mask_init_token,
            memory=obj_memory,
            pos=obj_memory_pos
        )
    mask_init_token = rearrange(mask_init_token, 'm b c -> b m c')

    # 拼回完整的 tokens
    output_tokens = cat([post_tokens, mask_init_token], dim=1)
    tokens = cat([output_tokens, sparse_prompt_embeddings], dim=1)
    # tokens: [B, 1+4+N, 256]

    # 标准 TwoWayTransformer
    src = image_embeddings + dense_prompt_embeddings  # image + mask prompt
    hs, src = self.transformer(src, pos_src, tokens)
    # → mask_tokens_out → output_hypernetworks → masks
```

## Sa2VA

2025, Text+VOS 类

### Motivation

- 引入 MLLM 来 prompt SAM2 以理解复杂的指代文本

### Pipeline

{% asset_img Sa2VA.png %}

### Prompting SAM2

`[SEG]` token 经过投影层后直接拼接 `sparse_pe` 注入 SAM2：

```python
extension/sam2_base.py:197-208：
sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
    points=(sam_point_coords, sam_point_labels),  # dummy point (0,0) label=-1
    boxes=None,
    masks=sam_mask_prompt,
)  # sparse_embeddings: [B, 2, 256]  ← 2 个 dummy not_a_point_embed

# Extension: LLM prompt
if language_embd is not None:
    assert sparse_embeddings.size(2) == language_embd.size(2)  # 256 == 256
    sparse_embeddings = torch.cat([sparse_embeddings, language_embd], dim=1) 
       # [B, 2, 256] + [B, N_obj, 256] → [B, 2+N_obj, 256]
```

## SIM-TOKEN

ICASSP 2026, Ref-AVS

### Motivation

- 希望引入 MLLM 增强理解能力
- MLLM 上下文窗口有限，需要压缩输入 token 数

### Pipeline

{% asset_img SIMTOKEN.png %}

### Prompting SAM

SIM-TOKEN 没有使用 Crab 的 Q-Former 方案压缩 LLM 输入，而是直接对视觉特征分别进行时间、空间维度上的平均池化得到 $F_{vt}, F_{vs}$，以及首帧的完整特征 $F_{vf}$ 与音频特征 $F_a$ 一起输入 LLM。得到 `[SEG]` token 后，直接拼接到每帧的 `sparse_pe` 上输入 SAM。SIM-TOKEN 尚未使用 SAM2，所以没有 tracking 和 memory，但是直接换成 SAM2 同理。

特别的，为了保证 `[SEG]` 不发生语义漂移，作者对同一目标的不同指代文本分别生成 `[SEG]` token，计算余弦相似度分布后用交叉熵损失向均匀分布拉平，以对齐 `[SEG]` token 语义。代码如下：

```python
def compute_alignment_loss(q, pos_feats, neg_feats, temperature=0.07):
    # q: [B, D] — 当前帧的 SEG token embedding
    # pos_feats: 同一物体在其他帧的 SEG token (存储于 SemanticMemoryBank)
    # neg_feats: 同视频中其他物体的 SEG token
    anchor = F.normalize(q[i])
    pos_tensors = F.normalize(torch.stack(pos))
    sim_pos = torch.matmul(anchor, pos_tensors.T) / temperature
    loss = -F.log_softmax(sim_pos).mean()
    # 使得同一对象在不同帧/不同表达式下的 <SEG> embedding 一致
```

## TGS-AGENT

AAAI 2026, Ref-AVS

### Pipeline

{% asset_img TGS-AGENT.png %}

### Prompting SAM2

TGS-Agent 使用 MLLM 基于复杂指代文本重新生成明确的目标标签，Grounding-DINO 基于标签得到目标的 bounding box，作为 SAM2 的 prompt 进行分割。利用了 MLLM 的理解能力，Grounding-DINO 文本到 bounding box 的强大能力，同时 box 是 SAM2 的 prompt encoder 天然支持的输入，没有特征对齐的问题；但是受限于标签语义和 Grouding-DINO 的特点，在多个物体的场景表现一般。

## SAM3-I

Ref-AVS

### Motivation

- SAM3 不擅长理解属性、空间位置、复杂指代等
- 当前方法多依赖外部 LLM，沉重且对不同实例区分度差

### Pipeline

{% asset_img SAM3-I.png %}

### Adapter

虽然 SAM3 的文本编码器是基于名词短语进行预训练，但是基础模型是读过丰富的各种文本的，所以也有理解较复杂文本的能力。对于涉及世界知识或者常识推理的指代文本，LLM 必须使用；但是大部分指代并不涉及这些内容，所以 SAM3 的文本编码器结合精心设计的 Adapters 也可能理解。作者设计了一组级联 Adapter：S-Adapter 和 C-Adapter，插入到 Text Encoder 和 Detector 中作为支路，分别用于处理简单和复杂问题。简单问题只使用 S-Adapter；复杂问题先经过 S-Adapter，后经过 C-Adapter。整个训练过程中，只有 Adapters 解冻，SAM3 主干模型全部冻结。

Adapter 采用瓶颈结构，下采样+GeLU+上采样。具体代码实现如下：

```python
# 变体 1（Encoder/Decoder 用） — encoder.py:251-292 / decoder.py:189-223：
class MultiHeadMLPAdapter(nn.Module):
    def __init__(self, d_model, adapter_dim=64, ...):
        self.down_proj = nn.Linear(d_model, adapter_dim)  # 例如 256→64
        self.up_proj = nn.Linear(adapter_dim, d_model)  # 64→256
        self.norm = nn.LayerNorm(adapter_dim)
        self.activation = nn.GELU()
        # 全部零初始化

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.up_proj(x)  # zero_init → 初始输出=0
        if self.skip_connect:
            return x + residual  # adapter2 用
        return x  # adapter1 / adapter12 不用

# 变体 2（Text Encoder 用） — text_encoder_ve_ada.py:133-200：
class MultiHeadMLPAdapter(nn.Module):
    def __init__(self, d_model, adapter_dim=64, num_heads=8):
        self.mha_adapter = nn.MultiheadAttention(d_model, num_heads)  # ← 多了 MHA
        self.down_proj = nn.Linear(d_model, adapter_dim)
        self.up_proj   = nn.Linear(adapter_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        # 1) MHA adapter (Pre-Norm, residual)
        x = x + self.mha_adapter(self.norm1(x))[0]
        # 2) Bottleneck MLP (Pre-Norm, residual)
        x = x + self.up_proj(self.activation(self.down_proj(self.norm2(x))))
        return x
```

每个添加了 Adapter 的层会创建三个实例：

```python
# encoder.py:370-391  /  decoder.py:276-292  /  text_encoder_ve_ada.py:57-73
self.adapter1  = MultiHeadMLPAdapter(...)  # 简单指令用
self.adapter12 = MultiHeadMLPAdapter(...)  # 级联第一阶段
self.adapter2  = MultiHeadMLPAdapter(..., skip_connect=True)  # 级联第二阶段，有残差
```

路由方式为：

```python
# encoder.py:396-417  /  decoder.py:297-317  /  text_encoder_ve_ada.py:75-99
def adapter_forward(self, x, text_type):
    if text_type == 1:  return x  # bypass
    elif text_type == 2:  out = self.adapter1(x)  # 简单指令
    elif text_type == 3:  out = self.adapter2(adapter12(x).detach())  # 复杂(冻结)
    elif text_type == 4:  out = self.adapter2(adapter12(x))  # 复杂(全梯度)
    return x + scale * out   # (encoder/decoder 有 scale+residual, text_encoder 无)
```

在文本编码器中，Adapter 插入在 Transformer Block 的 MLP 输出后，替代了残差连接。代码如下：

```python
text_encoder_ve_ada.py:111-130（AdapterResidualBlock.forward）：

def forward(self, q_x, k_x=None, v_x=None, attn_mask=None, text_type=None):
    # 标准 Attention + residual
    attn_output = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, ...)
    x = q_x + self.ls_1(attn_output)  # [B, L, 1024]
    # 标准 MLP
    mlp_output = self.mlp(self.ln_2(x)) # [B, L, 1024]
    if self.use_adapter and text_type is not None:
        # 在 MLP 输出处插入 adapter 输出，替代原始 skip connection
        x = self.adapter_forward(x, text_type)  # adapter 输出
        x = x + self.ls_2(mlp_output)  # + MLP 输出
        return x
```

在 Detector 中，Adapters 插入了 Encoder 和 Decoder 的每一层，并且都是替代了 FFN 的残差连接：

```python
# encoder.py:455-465（TransformerEncoderLayer_Ada.forward_post）：
tgt = tgt + dropout1(self.self_attn(tgt))  # Self-attn + residual
tgt = tgt + dropout2(self.cross_attn_image(tgt, memory))  # Cross-attn to image + residual
tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))  # FFN
if self.use_adapter and text_type is not None:  # Adapter 在 FFN 残差处插入
    tgt = self.adapter_forward(tgt, text_type) + self.dropout3(tgt2)  # adapter 和 FFN 输出相加作为新的 tgt
else:
    tgt = tgt + self.dropout3(tgt2)  # 原始：仅 FFN 残差
    
# decoder.py:323-335（TransformerDecoderLayer_Ada.forward_ffn）：
def forward_ffn(self, tgt, text_type=None):
    tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
    if self.use_adapter and text_type is not None:
        tgt = self.adapter_forward(tgt, text_type) + self.dropout4(tgt2)
    else:
        tgt = tgt + self.dropout4(tgt2)
    tgt = self.norm3(tgt)
    return tgt

# decoder 的完整 forward() 在 decoder.py:337-443：
def forward(self, tgt, ..., text_type=None):
    # self-attention on object queries + residual
    tgt = tgt + self.dropout2(self.self_attn(tgt))
    tgt = self.norm2(tgt)
    # cross-attention to text memory (text_attention_mask conditioned)
    tgt = tgt + self.catext_dropout(self.ca_text(tgt, memory_text))
    tgt = self.catext_norm(tgt)
    # cross-attention to image features
    tgt = tgt + self.dropout1(self.cross_attn(tgt, memory=image_feat))
    tgt = self.norm1(tgt)
    # FFN + adapter (同 encoder)
    tgt = self.forward_ffn(tgt, text_type=text_type)
    return tgt
```

#### 对比学习

为了让架构改动新增的模块对齐到 SAM3 的原始权重，设计了多种对比 loss：

- 指令对比损失：同一对象的简单/复杂指令 text embed 对齐；
- 亲级排名损失：复杂指令 embed 与所有概念做排名对比损失；
- 掩码分布对齐损失：计算同一对象简单/复杂指令 mask 的 KL 散度；
- 自适应权重：基于简单/复杂预测的差距计算不确定性图（难度大的不确定性高）作为 BCE 的权重。

还设计了三个训练阶段：

1. 冻结 SAM3，不使用复杂数据，单训 S-Adapter
2. 冻结 SAM3 和 S-Adapter，单训 C-Adapter；
3. 联合训练 Adapters，引入对比学习 loss 及权重

## Tarot-SAM3

### Motivation

- SAM3 缺乏复杂指代理解，需要 MLLM，但目前方法 MLLM 与 SAM3 结合的不好
  - 目前方法依赖 MLLM 给出 [SEG] token，若 MLLM 分割错了，SAM3 只会照着分割，没有纠错
  - SAM3 对不同名词很敏感，即便语义相同，词的选择不同也会影响表现
  - MLLM 可以参与 SAM3 输出 mask 的 refine，不断微调提示点让 SAM3 重新抠图
- 目前方法依赖在大型数据集上训练

### Pipeline

{% asset_img Tarot.png %}

### Architecture

```plaintext
输入: 图像 + 自然语言表达式 (e.g. "the player on the far right wearing a blue jersey")
    │
    ▼
┌──── Phase 1: ERI (Expression Reasoning Interpreter) ──────────────────┐
│                                                                       │
│  结构化推理 (RPO)                                                       │
│    MLLM 在 6 维结构化模板 T_R 下推理:                                    │
│      - 显式/隐式识别                                                    │
│      - 单/多目标判定                                                    │
│      - 指代对象分析                                                     │
│      - 形容词提取                                                       │
│      - 对象级推理                                                       │
│      - 分割混淆感知                                                     │
│    │                                                                  │
│    ▼                                                                  │
│  提取 N_t (目标名称) + N_r (相关参照对象集合)                              │
│    │                                                                  │
│    ▼                                                                  │
│  文本 Prompt 增强: N_t → {P_T^a, P_T^b, P_T^c} 三组同义表达              │
│    │                                                                  │
│    ▼                                                                  │
│  评估标准构建: N_r → SAM3 → bbox → MLLM 推理 → P_C^a (空间/语义关系谓词)    │
│    │                                                                   │
│    ▼                                                                   │
│  评估感知改写: P_C^a → T_S (简短) + T_L (含关系上下文的长版本)               │
│    T, T_S, T_L → MLLM 生成 3 个 bbox: {B_T^{Ini}, B_T^S, B_T^L}         │
│    │                                                                   │
│    ▼                                                                   │
│  SAM3 多类型生成:                                                        │
│    - text prompts → {M_Text^i}  (3 masks)                              │
│    - bbox prompts → {M_BBX^j}   (3 masks)                              │
│    │                                                                   │
│    ▼                                                                   │
│  Prompt-Consistency Filtering:                                         │
│    IoU(M_Text, M_BBX) < 0.80 → 丢弃                                     │
│    │                                                                   │
│    ▼                                                                   │
│  Per-type 最优: MLLM 评分 → M_Text^O, M_BBX^O                           │
│    │                                                                   │
│    ▼                                                                   │
│  DINOv3 引导 Point Prompt:                                              │
│    Ω = M_Text^O ∩ M_BBX^O (高置信重叠区)                                 │
│    从 Ω 采样锚点 → DINOv3 逐像素相似度图 S_D                               │
│    p_pos = argmax S_D within Ω                                         │
│    p_neg = argmax(-∇S_D · d_pos) outside Ω, S_D < 0.30                 │
│    → SAM3 → M_Point^O                                                  │
│                                                                        │
│  输出: {M_Text^O, M_BBX^O, M_Point^O}                                   │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──── Phase 2: MSR (Mask Self-Refining) ─────────────────────────────────┐
│                                                                        │
│  Inter-Prompt Selection (IPS):                                         │
│    MLLM 两两比较 {M_Text^O, M_BBX^O, M_Point^O} → 全局最优 M_Best^O       │
│    │                                                                   │
│    ▼                                                                   │
│  鉴别区构建:                                                             │
│    R_D^{Best}  = M_Best^O \ M_Point^O   (潜在过分割)                     │
│    R_D^{Point} = M_Point^O \ M_Best^O   (潜在欠分割)                     │
│    │                                                                   │
│    ▼                                                                   │
│  DINOv3 特征一致性:                                                      │
│    取 R_D 中心 → DINOv3 相似度图                                          │
│    R_D^{Best}:  区域内 min similarity → 可能不属于目标的子区域              │
│    R_D^{Point}: 区域内 max similarity → 可能遗漏的目标区域                 │
│    │                                                                   │
│    ▼                                                                   │
│  MLLM 归属判定:                                                          │
│    滤出的鉴别区 + 高置信区 → MLLM 判断是否属于同一目标                        │
│    │                                                                   │
│    ▼                                                                   │
│  点 Prompt 修正 (OPM):                                                  │
│    over-seg  → p_neg 移到 R_D^{Best} 中心                               │
│    under-seg → p_pos 移到/新增到 R_D^{Point} 中心                        │
│    │                                                                   │
│    ▼                                                                   │
│  修正 prompt + M_Best^O → SAM3 → 最终精炼 mask                           │
└────────────────────────────────────────────────────────────────────────┘
```

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\(', '\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>