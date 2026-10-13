# M0: Hub pins

Checked 2026-10-05, after `huggingface.co` was allowed in the session's network policy. This file closes the "needs a full download" items in `docs/m0/encoder-selection.md` section 5 and the Hub-dependent fields in `model-pipeline/admissions/kev.md`.

## Method

1. `GET https://huggingface.co/api/models/<repo>/revision/<rev>?blobs=true` resolved each revision to a full commit SHA and listed every file with its size, git blob ID, and (for LFS files) the SHA-256 the Hub stores.
2. Each file used by an M0 run was downloaded from `https://huggingface.co/<repo>/resolve/<commit SHA>/<path>`, never from a branch name, into `.models/` (ignored by Git).
3. SHA-256 and size were computed locally. Every downloaded file matched the Hub's size, and every LFS file matched the Hub's SHA-256. No mismatch was found.

Files not used by any M0 run were not downloaded. They are listed with the Hub's own record (LFS SHA-256, or git blob ID for small files) so the full repository content at the pinned commit is visible.

## Summary

| Repository | Pinned commit | Commit date (UTC) | License (source) | Used for |
| --- | --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | 2026-04-02 | MIT (card front matter; no LICENSE file) | P0 encoder working candidate: manifest, reference runner, fixtures |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` | 2026-01-28 | Apache-2.0 (card front matter; no LICENSE file) | Fallback encoder; tokenizer and config files only |
| `fastino/gliner2-base-v1` | `f9634218e53580c56edf0de97ca1a7d3f1c2354e` | 2026-09-28 | Apache-2.0 (card front matter) | GLiNER2 tokenizer fixtures; weights not downloaded |
| `microsoft/deberta-v3-base` | `8ccc9b6f36199bec6961081d44eb72fb3f7353f3` | 2022-09-22 | MIT (card front matter) | Backbone `spm.model` check |
| `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | 2026-04-23 | Apache-2.0 (`LICENSE` file) | Kev-0.8B base weights and tokenizer |
| `jaredpalmer/kev-0.8b` | `9a45d25eb2ab761841196625383fa1dff0e56c1e` | 2026-09-24 | Apache-2.0 (card front matter; no LICENSE file) | Kev spike (P-027) |

Branch heads on the check date: `jaredpalmer/kev-0.8b` `main` is `bf75a6a8848ea6960ff2ed108d9ed44c2941174f` ("Kev 1.0 model card (weights unchanged)"). The Hub tag `v1.0` resolves to the same commit (its ref lists the annotated tag object `788ddbdd65715bb03a56788c822f6c632c9a551d`). Every weight, head and tokenizer file at `bf75a6a8` has the same LFS SHA-256 and git blob ID as at `9a45d25e`; only `README.md` differs. `Qwen/Qwen3.5-0.8B-Base` `main` is `dc7cdfe2`, the revision Kev records. `fastino/gliner2-base-v1` `main` moved to `f9634218` on 2026-09-28, after the GLiNER2 source commit inspected in M0 (2026-09-24).

Fallback revisions, recorded without download:

| Repository | `main` commit | Commit date (UTC) | Files of note (Hub LFS SHA-256) |
| --- | --- | --- | --- |
| `jaredpalmer/kev-0.6b` | `dece6dba8d43f0f7ded45e9f5b9df12474d90843` | 2026-09-20 | `adapter_model.safetensors` `deaab63b61f95d628503831e8b336e1c3e334b9a42d3530d979a5149b429c492` (40,419,816), `head.pt` `ce6cd9ffc54db41c179b65a33d60973dc8280c29e886218eb3ec07dc28d6f28b` (2,102,399), `tokenizer.json` `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` (11,422,654) |
| `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | 2025-07-26 | `model.safetensors` `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba` (1,192,135,096) |

## Answers to the open questions

- **Which `tokenizer.json` Kev loads.** Kev calls `AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base", revision="dc7cdfe2...")` (`kev/model.py` `load_tokenizer`). Under transformers 5.17.0 that returns `Qwen2Tokenizer`, because the base `tokenizer_config.json` names that class. transformers 5 builds the pipeline from the class, not from the base `tokenizer.json`, and the live pipeline (`backend_tokenizer.to_str()`) is equal, field by field, to the Kev repository's `tokenizer.json` (`06b95093...`, 19,989,325 bytes): the Qwen2/Qwen3 regex without `\p{M}`, ByteLevel `trim_offsets` true, 33 added tokens. Vocabulary and merges match the base file (`fe000e3e...`, 12,807,196 bytes); the base file's `\p{M}` regex and its 22 added tokens are not used at runtime. On the M0 corpus the two pipelines give different ids on 2 of 46 cases (`devanagari`, `thai`, both with combining marks). The upstream release record's `tokenizer.json` hash `06b95093...` is the Kev repository copy, which is also the effective runtime tokenizer. The owned Qwen BPE profile for Kev should target that file.
- **Which `head.pt` is the release head.** `head.pt` at `9a45d25e` (and at `bf75a6a8`) is `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6`, the hash in the upstream release record. Loaded with `torch.load(weights_only=True)`, it carries `temperature` 2.3510958125672174 and a `temperature_fit` entry (121-point grid fit on decision-v7 development rows). The `bc77488a...` hash in the Hub `provenance.json` (temperature 1.0) is not the file on the Hub at either commit. It is consistent with `head.pt` as the training run wrote it, before the temperature fit was stored in the same file; that is an inference from the file's `temperature_fit` entry and the provenance temperature, not verified by bytes.
- **`onnx/tokenizer.json` against `tokenizer.json` (e5).** Identical: same SHA-256 `0b44a9d7...`. `onnx/sentencepiece.bpe.model` is also identical to the root copy.
- **DeBERTa-v3 vendored hashes.** `microsoft/deberta-v3-base` `spm.model` is `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd`, equal to the vendored hash in `encoder-selection.md` section 2, and `fastino/gliner2-base-v1` ships the same `spm.model`. The base repository has no `tokenizer.json`, so the vendored xsmall `tokenizer.json` hash (`05402ffa...`) has no base counterpart; GLiNER2 ships its own `tokenizer.json` (`1b7fbabf...`) with its 10 added tokens. The findings that relied on the vendored `spm.model` stand.
- **e5 and MiniLM share the SentencePiece model.** Both repositories ship `sentencepiece.bpe.model` `cfc8146a...`, the same file as the vendored XLM-R copy in `encoder-selection.md` section 3. Their `tokenizer.json` files differ (`0b44a9d7...` against `2c3387be...`).

## Files

### `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`

Downloaded and hashed:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `1_Pooling/config.json` | 200 | `987f7a67a38fa564c849bb5d277c52ab9088a84368fc0be31a354125aebb12a0` |
| `README.md` | 497,538 | `0038de97aee16258cecbad7ffda4b4febd6953e747a00e0ddbc8e6ed241e9c1c` |
| `config.json` | 655 | `69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959` |
| `model.safetensors` | 470,641,600 | `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477` |
| `modules.json` | 387 | `c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04` |
| `onnx/config.json` | 653 | `bbb7c1333fc4b3e27fbc9cd5d2070aabcc1d4dfb99917c3633e772f97545a6b6` |
| `onnx/model.onnx` | 470,268,510 | `ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665` |
| `onnx/sentencepiece.bpe.model` | 5,069,051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `onnx/special_tokens_map.json` | 167 | `d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7` |
| `onnx/tokenizer.json` | 17,082,730 | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39` |
| `onnx/tokenizer_config.json` | 443 | `a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b` |
| `sentence_bert_config.json` | 57 | `948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529` |
| `sentencepiece.bpe.model` | 5,069,051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `special_tokens_map.json` | 167 | `d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7` |
| `tokenizer.json` | 17,082,730 | `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39` |
| `tokenizer_config.json` | 443 | `a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b` |

Not downloaded (not used). LFS SHA-256 from the Hub API where the file is in LFS; git blob ID otherwise:

| File | Size (bytes) | Hub record |
| --- | ---: | --- |
| `onnx/model_O4.onnx` | 235,052,531 | SHA-256 `4654c156f3e4171abc9c716cdb771bf9116455d15ac1aab364aeeede0e3205b0` |
| `onnx/model_qint8_avx512_vnni.onnx` | 118,346,824 | SHA-256 `dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88` |
| `openvino/openvino_model.bin` | 470,027,936 | SHA-256 `22f5a10ec005a1606ec42758dedd52b59dfcb6e3d85de0b6b86946d0bc6cf232` |
| `openvino/openvino_model.xml` | 363,385 | git blob `141b2bf7854f27e1b3369fc19c6377bb799bb215` |
| `pytorch_model.bin` | 470,681,649 | SHA-256 `f0ccb64a2edc689fce7b1895f25146740e4075c3da28c8d0e672a8ba49c51167` |

21 `.eval_results/*.yaml` files (MTEB result cards) not downloaded or listed.

### `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` at `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`

Downloaded and hashed:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `1_Pooling/config.json` | 190 | `4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23` |
| `README.md` | 3,888 | `1e98ea05b0de579fcaad3d625b62ea55647142ed674d5f5ebf1440e4bbbb6f23` |
| `config.json` | 645 | `6300193cb75e01cf80c96decef7187dfb33094d97cc1490b7ead6ff134476e4e` |
| `config_sentence_transformers.json` | 122 | `b8c64b5cece00d8424b4896ea75b512b6008576088497609dfeb6bd63e6d36b8` |
| `modules.json` | 229 | `8f4b264b80206c830bebbdcae377e137925650a433b689343a63bdc9b3145460` |
| `sentence_bert_config.json` | 53 | `70f4448f31320443fe3557cacea5abf2dcc4915dda8c80646bec9f3bb0aa5a1f` |
| `sentencepiece.bpe.model` | 5,069,051 | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |
| `special_tokens_map.json` | 239 | `378eb3bf733eb16e65792d7e3fda5b8a4631387ca04d2015199c4d4f22ae554d` |
| `tokenizer.json` | 9,081,518 | `2c3387be76557bd40970cec13153b3bbf80407865484b209e655e5e4729076b8` |
| `tokenizer_config.json` | 526 | `5036ea374ffedd706e3bef33e2e0d6953cb868ef8a490e76e32ba0faa37a6b9b` |

Not downloaded (not used). LFS SHA-256 from the Hub API where the file is in LFS; git blob ID otherwise:

| File | Size (bytes) | Hub record |
| --- | ---: | --- |
| `model.safetensors` | 470,641,600 | SHA-256 `eaa086f0ffee582aeb45b36e34cdd1fe2d6de2bef61f8a559a1bbc9bd955917b` |
| `onnx/model.onnx` | 470,301,610 | SHA-256 `10f7a088420252b26caf819236ca2c9d2987afd0fc06fec7553b542a5655a05a` |
| `onnx/model_O1.onnx` | 470,212,363 | SHA-256 `9ae4b831e992807334f18a91557661e94715f502a5c7248fb81675b08391e30f` |
| `onnx/model_O2.onnx` | 470,145,917 | SHA-256 `338ef03c2838d5a659d36e1ce5b7a1dc2d2a66a430a9e6f499de6dc39f663850` |
| `onnx/model_O3.onnx` | 470,145,772 | SHA-256 `2426785672da5afa2ca3ac1efae652499c72a3198e8e203d06f6d6e6c569d419` |
| `onnx/model_O4.onnx` | 235,166,264 | SHA-256 `307bba13f9f5708461169c9f2d633c76c5572919bcc998606b6e5aea46f05db4` |
| `onnx/model_qint8_arm64.onnx` | 118,412,398 | SHA-256 `783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672` |
| `onnx/model_qint8_avx512.onnx` | 118,412,398 | SHA-256 `783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672` |
| `onnx/model_qint8_avx512_vnni.onnx` | 118,412,398 | SHA-256 `783fea82d71a58179b830a4dbd2d58447e640609e98eedf9ffa12622d375a672` |
| `onnx/model_quint8_avx2.onnx` | 118,453,870 | SHA-256 `98a01d88b7de996cdea58c32ca71208c09968d143798814b2ea09d3439dc334f` |
| `openvino/openvino_model.bin` | 470,027,920 | SHA-256 `04e8cc4ceedb65316f374f798b9428b491281e064b4cb6076e6abf0221256ac1` |
| `openvino/openvino_model.xml` | 398,501 | git blob `5e34ccc5b60aa53a44488b75b78c5d4f9c0b7cc2` |
| `openvino/openvino_model_qint8_quantized.bin` | 118,989,868 | SHA-256 `24acd56a5f5ae4ba5b39c9593997ebb6d5da44a6439ef1d0757a70c70aadb7e3` |
| `openvino/openvino_model_qint8_quantized.xml` | 708,844 | git blob `d4c56fb65c7171dc9c41c17f556ec5d51f219416` |
| `pytorch_model.bin` | 470,693,617 | SHA-256 `16cc9e54df6e083272378abec2d75dc34d7a48b5276db3ccc050d18de672ac59` |
| `tf_model.h5` | 470,899,176 | SHA-256 `22150b6ba00e477c7f816f1988d028fff924e2b52e14540889690c72c5add40e` |
| `unigram.json` | 14,763,234 | SHA-256 `71b44701d7efd054205115acfa6ef126c5d2f84bd3affe0c59e48163674d19a6` |

### `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`

Downloaded and hashed:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `README.md` | 6,784 | `ec5c9215186974ec78e851af8894a5d433e1205a33f8e1dca3f49a70a159c30a` |
| `added_tokens.json` | 230 | `3c274ca559be8b7b18febf616724ec05f8a8abccd428bd2744608f45f7a0b067` |
| `config.json` | 236 | `33d8aa009deed4b5435d162e08b5a720b252a0084e59d30d366b37da8c0193c5` |
| `encoder_config/config.json` | 823 | `9840a4db70bc007e6b65d336ebe2bddc53bc2ce210dc5757e50d5bb17122f7cd` |
| `special_tokens_map.json` | 2,414 | `84ea70143f533d7e99b393d87f20010887a9ac2cba955828ef313886e4e83f4f` |
| `spm.model` | 2,464,616 | `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd` |
| `tokenizer.json` | 8,650,701 | `1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16` |
| `tokenizer_config.json` | 3,240 | `4e6c6a8d26478b670692916266438485403b0c54af31fd1a12bab8a465d062e9` |

Not downloaded (not used). LFS SHA-256 from the Hub API where the file is in LFS; git blob ID otherwise:

| File | Size (bytes) | Hub record |
| --- | ---: | --- |
| `GitHub_new.jpg` | 858,002 | SHA-256 `428f13aa08f1aa12370c4a644f235ddc46db85e429a1dee6a8d22ead095f2183` |
| `SKILL.md` | 6,511 | git blob `5990b5ca35d31b9ec71f013095feb5789fe53147` |
| `image/GLiNER-2-Base-HF-Banner.png` | 654,344 | SHA-256 `b1bbdff39b4024a3ebdcd11520c26da5f1fb29c987d0b539341ecb652af91eba` |
| `image/GitHub.png` | 4,408,751 | SHA-256 `c945cbd7b6178aa0e80ed2f10382d6edf59799e3e0dd9b24e3e8c5fe58c02deb` |
| `model.safetensors` | 833,938,108 | SHA-256 `845fc4bd93c525b86124c58ab4f56c9eacf8587953086b14c501fab25957c007` |

### `microsoft/deberta-v3-base` at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`

Downloaded and hashed:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `README.md` | 3,472 | `7e9ea69ca00d0609cc0e0b9cec76f9ba6ae71601244c4136a3f25bd02dce4c4d` |
| `config.json` | 579 | `649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b` |
| `spm.model` | 2,464,616 | `c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd` |
| `tokenizer_config.json` | 52 | `3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d` |

Not downloaded (not used). LFS SHA-256 from the Hub API where the file is in LFS; git blob ID otherwise:

| File | Size (bytes) | Hub record |
| --- | ---: | --- |
| `pytorch_model.bin` | 371,146,213 | SHA-256 `691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33` |
| `rust_model.ot` | 742,218,621 | SHA-256 `276aadc323988709f076fb489790103d28e64b80c72e9a3e19043d28f4c7c31a` |
| `tf_model.h5` | 735,589,384 | SHA-256 `01ad1b35cac509fb00b9873c670d824363ef884d1aa2758471c47b26cc2948f0` |

### `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`

Downloaded and hashed:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `LICENSE` | 11,343 | `50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32` |
| `README.md` | 3,727 | `3cff39d2674e28b95e2df75cdb2d7a687ec391931065b05df83dbae12b447b11` |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c` |
| `model.safetensors.index.json` | 50,900 | `ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer.json` | 12,807,196 | `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927` |
| `tokenizer_config.json` | 16,712 | `e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de` |
| `video_preprocessor_config.json` | 386 | `d039cd7d88b3502a99edd455496b75b44ecf7dd3b3669748bafac37e6cecc085` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

### `jaredpalmer/kev-0.8b` at `9a45d25eb2ab761841196625383fa1dff0e56c1e`

Downloaded and hashed:

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `README.md` | 18,516 | `5c0408069bf3882c758ccc10a8351a4fba6a66e5cdce0ae6b3c320f4a50c0afb` |
| `adapter_config.json` | 1,273 | `748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854` |
| `adapter_model.safetensors` | 43,338,624 | `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a` |
| `head.pt` | 2,103,999 | `f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6` |
| `provenance.json` | 4,221 | `fce039fbda7de19451c9b5d3065aee2c5e063e1562a8bc18c72e297fbd15e9d8` |
| `result.json` | 80,760 | `523770dbe04ad23bfdadd627d253273220fe8969a74b213e0fafb1e65093bc8c` |
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | 1,128 | `8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45` |
| `train.log` | 18,427 | `70483db425283555737ea9818a13910de767375e63febd6648023deb3fcaccfd` |
| `training_config.json` | 1,767 | `3ec3d92146248265652d4387f8eb65ee8defaed4dd9173b049e7b5193045ece6` |
| `training_metrics.json` | 326 | `5f94d869c3707c0efff9f2e893e288014bc88515ae04fe5f75c9d793c453cfc1` |


## Pilot LLM: `Qwen/Qwen3.5-0.8B`

Added 2026-10-05 for the generation-backend check (`docs/m0/generation-backend-check.md`). `Qwen/Qwen3.5-0.8B` is the chat (post-trained) model; the Hub lists only it and `Qwen/Qwen3.5-0.8B-Base` under that name in the `Qwen` namespace. `main` resolved to `2fc06364715b967f1860aea9cf38778875588b17` (commit date 2026-03-02); the repository has no tags. License: Apache-2.0 (card front matter and `LICENSE` file). Card pipeline tag: `image-text-to-text`; `config.json` declares `Qwen3_5ForConditionalGeneration` with a 12-layer vision tower, and the check uses the text decoder only.

Every file was downloaded by commit SHA with `model-pipeline/reference/generation/fetch_snapshot.py`, which checks size against the Hub and, for LFS files, the Hub's SHA-256. All matched.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `.gitattributes` | 1,570 | `34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930` |
| `LICENSE` | 11,544 | `bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a` |
| `README.md` | 61,705 | `87a163af54f32fa608a0f8d3ac67945c53dd2b4a7c96740b3d7fdc28e8458864` |
| `chat_template.jinja` | 7,755 | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `config.json` | 2,907 | `b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204` |
| `merges.txt` | 3,353,259 | `a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d` |
| `model.safetensors-00001-of-00001.safetensors` | 1,746,942,600 | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |
| `model.safetensors.index.json` | 50,900 | `d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e` |
| `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `tokenizer.json` | 12,807,982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| `tokenizer_config.json` | 16,709 | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |
| `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `vocab.json` | 6,722,759 | `ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003` |

Relation to `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2` (the Kev base):

- Same bytes: `config.json`, `merges.txt`, `vocab.json`, `preprocessor_config.json`. Different: the weights, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `video_preprocessor_config.json`, `LICENSE`, `README.md`.
- `chat_template.jinja` and the `chat_template` field of `tokenizer_config.json` are the same string. The template renders tools, `reasoning_content`, and appends `<think>\n\n</think>\n\n` after the generation prompt unless `enable_thinking` is true. `tokenizer_config.json` sets `eos_token` `<|im_end|>` (248046), `pad_token` `<|endoftext|>` (248044), `add_bos_token` false; `text_config.eos_token_id` is 248044. There is no `generation_config.json`.
- `tokenizer.json` has the Qwen3.5 Split regex with `\p{M}`, ByteLevel `trim_offsets` false, and 26 added tokens (248044 to 248069); vocabulary and merges equal the base file's.
- Runtime tokenizer: transformers 5.18.0 loads both the chat and the base snapshot as `Qwen3_5Tokenizer` (although both `tokenizer_config.json` files name `Qwen2Tokenizer`). Its live pipeline keeps the `\p{M}` regex, sets ByteLevel `trim_offsets` true, uses a ByteLevel decoder with `add_prefix_space` and `use_regex` true, and has 33 added tokens (adds 248070 to 248076, the audio and TTS specials). The ONNX Runtime GenAI builder writes that pipeline as `tokenizer.json` (`a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f`, 19,989,339 bytes). Under transformers 5.17.0 (Kev's pinned environment) the same base snapshot loads as `Qwen2Tokenizer` without `\p{M}` (the Kev repository `tokenizer.json`, `06b95093...`, see above). The two runtime pipelines differ only in the `\p{M}` regex, so they give different ids on text with combining marks. Which one the owned Qwen BPE profile targets is an open decision (generation-backend check, section 9).


## P1 embedding candidate: `google/embeddinggemma-2`

Added 2026-10-10 for the EmbeddingGemma 2 admission spike (`model-pipeline/admissions/embeddinggemma-2.md`). Pinned commit `914f7f89142e33e77833254d9c9b90c3cef7303b` (Hub `main` on 2026-10-10; repository created 2026-09-14, last modified 2026-10-06; not gated). License: Apache-2.0 (card front matter `license: apache-2.0`; no LICENSE file in the repository). Card pipeline tag `feature-extraction`; `config.json` declares `EmbeddingGemma2Model` with text, vision and audio towers, and the spike uses the text path only.

Code: transformers 5.19.0 (PyPI, uploaded 2026-10-06) is the first release that defines `embedding_gemma2`; 5.16.1, 5.17.0 and 5.18.0 do not (checked in the wheels). The checkpoint's `config.json` names `transformers_version` `5.18.0.dev0`, a development build. Wheel SHA-256 `afcd2dd5f603ed28c1e1fcb00a338ccbb4ef5f878ed289635df8b58187afb518`. sentence-transformers 6.1.0 (the version `config_sentence_transformers.json` records and requires, `>=6.1.0`). Both are pinned with hashes in `model-pipeline/reference/embeddinggemma2/requirements.txt`.

Every file in the repository was downloaded from `https://huggingface.co/google/embeddinggemma-2/resolve/914f7f89142e33e77833254d9c9b90c3cef7303b/<path>` into `/home/mark/Projects/AI-models/embeddinggemma2/snapshot/` (outside Git). Sizes match the Hub API; the three LFS files match the Hub's SHA-256, and the git blob ID of every small file, recomputed locally, matches the Hub's.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `.gitattributes` | 1,301 | `bcdccfaa66bc1240c7192dbd374cfb39b695bb1d8ff6b7ce0113aa5e9ac21a44` |
| `1_Pooling/config.json` | 90 | `8759bdf7c77efc7df7723f64856a593c8943b71ee38baf2a88771fbaf78438f9` |
| `2_Normalize/config.json` | 97 | `cdb09dfca347a56aa2d691744e38d5ad3c7cbc2834e7181272b9a15328b82524` |
| `README.md` | 21,730 | `b677a0ee2818c36f091b65fd3d9b4baea9f04d53b02b49cae916c74bf9682037` |
| `chat_template.jinja` | 1,016 | `4b852efc0b9960283e735363331e6f325b33bc74bdbaa076f595bc4e9b94d85e` |
| `config.json` | 4,455 | `b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3` |
| `config_sentence_transformers.json` | 1,565 | `031e56a498d33c349ab489a21885bcfe25b4fcba841149dc99e1e90d4a7c28f5` |
| `model.safetensors` | 1,488,915,288 | `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79` |
| `modules.json` | 413 | `3d02572a0455b832de67fb8e63a54981bc7e8b46e337c95e917bd8122a533bfd` |
| `preprocessor_config.json` | 511 | `ea2ae257e901064abdd98dceb19f2b0da06af600bed15e0f99f5c85c37ee9d78` |
| `processor_config.json` | 1,788 | `168f6a08522f3ce5dea596d94d003af2fd691742d4f41fe1f9d8cce76bfbf69c` |
| `sentence_bert_config.json` | 747 | `b1bcd9f2dce3ae863b359e87d0710b5dbc3314a59ecb4e2f97c7778fc8e4b228` |
| `tokenizer.json` | 32,170,510 | `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |
| `tokenizer.model` | 4,689,013 | `e594c8a90eb08d8bda498ff4747977dc827ae0c3c56b5c0d41a605a22d02ef03` |
| `tokenizer_config.json` | 1,599 | `17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874` |

Notes:

- `model.safetensors` is 1,488,915,288 bytes, bfloat16, all towers: 744,371,992 parameters (language model 271,002,648, vision tower 167,364,608, audio tower 304,825,088, multimodal embedders 1,179,648), read from the safetensors header. The provisional admission record's "744,371,512 bytes" was close to the parameter count, not the file size; corrected there. The text path has 271,002,624 trainable parameters (plus 24 one-element `layer_scalar` buffers stored in the checkpoint), about 1.08 GB in fp32.
- Runtime tokenizer: transformers 5.19.0 loads `GemmaTokenizer` (the class `tokenizer_config.json` names), and its live pipeline (`backend_tokenizer.to_str()`) is equal, field by field, to the repository `tokenizer.json`. On the M0 corpus the ids agree on 48 of 49 cases; the exception is `pair-empty-second`, where transformers drops an empty second sequence and the `tokenizers` library encodes it as `<bos><eos>`. sentence-transformers 6.1.0 renders text through `chat_template.jinja` (prompt as a system message, then the text) and gives the same ids as `tokenizer.json` on `prompt + text` for all 46 single texts with the three prompt settings tested. The owned profile should target `tokenizer.json` (`4d777ef5...`).
- `tokenizer.model` (SentencePiece) is not used by any loader at the pinned versions; recorded for completeness.
