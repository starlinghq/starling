# Proposed: EmbeddingGemma 2 image release fixture (P-030), Wikimedia Commons images

Status: **proposed**. This fixture is not accepted and is not under `model-pipeline/protected/`. Intended use: the held-out G1/G2 image release set for `embeddinggemma-2/image-fp32-cpu` and `image-fp32-cuda`. If accepted, the files move byte-identical to a protected release directory in a `protected-change` pull request. No .NET run has used these files.

## Media

The media are the 16 images of the Stable proposal's `embeddinggemma-2/media/media.json` (PR #20, SHA-256 `7501f65cfb6d2586305f9efe76f512babbaf999d8fb9301405d87db80b3b7198`). `select_media.py` drew them from `Category:Featured pictures on Wikimedia Commons` by seeded sample, and no agent chose any image. **No media bytes are committed.** `images.json` pins each file by its Commons URL, revision, SHA-1, SHA-256 and the SHA-256 of its decoded pixels, as the development images are pinned (`make_eg2_images.py`). The script below regenerates `images.json` and the files byte for byte:

```sh
python -I model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py --media <media.json> --out <media dir>
```

Before downloading, `fetch_eg2_release_media.py` re-checks every file. It queries the Commons API `extmetadata` (`LicenseShortName`, `License`, `Artist`, `Credit`, `UsageTerms`, `AttributionRequired`) and the license templates of the description page, and it checks that the current revision's SHA-1 and size equal the recorded ones. It keeps only public domain, CC0 and CC BY files and rejects any SA, NC, ND or GFDL term. The check ran on 2026-10-14 (`licenses.json`). **All 16 images passed; none was rejected.** Licenses as Commons states them (data, not a legal review; the maintainer confirms each on its description page):

| Id | File (description page) | Revision | License (page templates) | Author / credit (Commons `Artist`) | Bytes | SHA-256 |
| --- | --- | --- | --- | --- | ---: | --- |
| `commons-image-00` | [A New View of the Tarantula Nebula.jpg](https://commons.wikimedia.org/wiki/File:A_New_View_of_the_Tarantula_Nebula.jpg) | 2026-06-29T23:07:11Z | Public domain (PD-USGov-NASA) | NASA | 9,279,797 | `6f332c508cb0e22da3b39fb46a8e8d965553a8437f69c6e58f7e22dd4a13a42a` |
| `commons-image-01` | [A Wilde time 3.jpg](https://commons.wikimedia.org/wiki/File:A_Wilde_time_3.jpg) | 2009-12-13T05:45:47Z | Public domain (Cc-pd-mark-footer, PD-US-expired-text, PD-old-X-expired, PD-old-auto-expired, PD-old-text) | Napoleon Sarony | 9,633,730 | `dbb51633b1eb5ba7a09134a57426d3925f5bbbfba7f666043b2e9a8d7b82a123` |
| `commons-image-02` | [A member of the ATS (Auxiliary Territorial Service) serving with a 3.7-inch anti-aircraft gun battery, December 1942. TR452.jpg](https://commons.wikimedia.org/wiki/File:A_member_of_the_ATS_(Auxiliary_Territorial_Service)_serving_with_a_3.7-inch_anti-aircraft_gun_battery,_December_1942._TR452.jpg) | 2017-10-08T21:51:01Z | Public domain (PD-Scan, PD-UKGov, PD-art-category, PD-scan) | Malindine E G (Lt), Tanner (Lt), War Office official photographer | 1,693,820 | `b71d76c9bdcb5dc69258842e443e5a29b59e0a1d3c9d7e50e0a758eb35f422ab` |
| `commons-image-03` | [A package for Hitler 196462 edit.jpg](https://commons.wikimedia.org/wiki/File:A_package_for_Hitler_196462_edit.jpg) | 2012-04-06T10:38:25Z | Public domain (PD-USGov-FSA) | Alfred T. Palmer | 3,233,346 | `89c0638f94ef6bb4513b765126ef092bdb40b31883a9c580b4354f6816f0571f` |
| `commons-image-04` | [A pair of little Harvest Mice (50331384776).jpg](https://commons.wikimedia.org/wiki/File:A_pair_of_little_Harvest_Mice_(50331384776).jpg) | 2023-10-24T13:33:49Z | CC BY 2.0 (Cc-by-2.0) | caroline legg | 2,664,407 | `943a6dd5ea7851754afcfeb191b432b8ecc343658b921450302c8ea07e2afb0b` |
| `commons-image-05` | [E 003261 E Maoris in North Africa July 1941.jpg](https://commons.wikimedia.org/wiki/File:E_003261_E_Maoris_in_North_Africa_July_1941.jpg) | 2016-02-24T09:53:42Z | Public domain (PD-NZ, PD-New Zealand, PD-old-warning-text) | Unidentified New Zealand official photographer; Restoration by Adam Cuerden | 2,907,723 | `f6255776bc5a6b9cd4248418670aedaa0cade6dc1aca431c096c4d064476f44c` |
| `commons-image-06` | [Earthset (art002e009288).jpg](https://commons.wikimedia.org/wiki/File:Earthset_(art002e009288).jpg) | 2026-04-07T13:07:40Z | Public domain (PD-USGov-NASA) | NASA | 884,095 | `1a172c85a5b7172cfffb7e3811056f22086b7dcfef56c369d5e6a66fb8a363ba` |
| `commons-image-07` | [M.Nonius Balbus Nuceria equestrian Statue MANN.png](https://commons.wikimedia.org/wiki/File:M.Nonius_Balbus_Nuceria_equestrian_Statue_MANN.png) | 2016-10-17T09:46:32Z | CC0 (Cc-zero) | Jebulon | 6,972,938 | `5311a11fba535763f098e9aa084e5dfebf7933a5699ca5638ea2cdd83025d4e6` |
| `commons-image-08` | [Maersk headquarters Copenhagen 2014 01.jpg](https://commons.wikimedia.org/wiki/File:Maersk_headquarters_Copenhagen_2014_01.jpg) | 2014-11-09T19:13:06Z | CC BY 4.0 (Cc-by-4.0) | Julian Herzog (Website) | 5,753,572 | `996700d3155dbaecada63987d0474dc6251fdd16c0d2b2bbe3f8f0d65c729ad2` |
| `commons-image-09` | [Magellanic penguin, Valdes Peninsula, e.jpg](https://commons.wikimedia.org/wiki/File:Magellanic_penguin,_Valdes_Peninsula,_e.jpg) | 2011-03-01T15:55:55Z | CC BY 2.0 (Cc-by-2.0) | David | 2,220,239 | `ceaf08822b1eee5ec3eab78b0d152ce360e27938dfcba520cc29cbea0c09458a` |
| `commons-image-10` | [Queen Victoria surrounded by her family - Coburg, 1894 (1 of 2).jpg](https://commons.wikimedia.org/wiki/File:Queen_Victoria_surrounded_by_her_family_-_Coburg,_1894_(1_of_2).jpg) | 2024-10-02T23:43:10Z | Public domain (Cc-pd-mark-footer, PD-US-expired-text, PD-old-X-expired, PD-old-auto-1923, PD-old-auto-expired, PD-old-text) | Eduard Uhlenhuth | 5,497,839 | `6627fe99fe461b9769683d754ff26881ae52370d81253b5b288830806c794e7b` |
| `commons-image-11` | [Raising a flag over the Reichstag - Restoration.jpg](https://commons.wikimedia.org/wiki/File:Raising_a_flag_over_the_Reichstag_-_Restoration.jpg) | 2024-03-16T08:43:11Z | Public domain (Cc-by-4.0, PD-Russia-1996) | Yevgeny Khaldei / Adam Cuerden | 5,926,198 | `3790d08cc52c373d77868e65e74fd4ce6e507671db39faefb3f1d9b5f7cc7a6a` |
| `commons-image-12` | [U.S. Navy T-6B Texans in a 4-ship formation over a military operations area in Southern Texas.jpg](https://commons.wikimedia.org/wiki/File:U.S._Navy_T-6B_Texans_in_a_4-ship_formation_over_a_military_operations_area_in_Southern_Texas.jpg) | 2021-03-12T20:25:15Z | Public domain (PD-USAF, PD-USGov-Military-Air Force) | Master Sgt. Christopher Boitz | 1,852,064 | `db8d8e050100e9d787a9df89441e558ba34715831cef63820bcfba7a38555986` |
| `commons-image-13` | [ULPower UL260i.jpg](https://commons.wikimedia.org/wiki/File:ULPower_UL260i.jpg) | 2008-02-16T15:54:35Z | Public domain (PD-author) | ULPower | 624,553 | `42832fd277f8b89574dee28d5765b52560ae89797f834ba7d6c1d443fc39a8d8` |
| `commons-image-14` | [Umnak oli 2014123 lrg.jpg](https://commons.wikimedia.org/wiki/File:Umnak_oli_2014123_lrg.jpg) | 2017-11-04T03:33:26Z | Public domain (PD-USGov-NASA) | NASA Earth Observatory images by Joshua Stevens, using Landsat data from the U.S. Geological Survey. Story by Mike Carlowicz. | 10,302,692 | `4f93b2279e0a7c6f478c599465c391ea50cc1e809cd2d809b755793757175ed9` |
| `commons-image-15` | [Underwater mcmurdo sound.jpg](https://commons.wikimedia.org/wiki/File:Underwater_mcmurdo_sound.jpg) | 2008-01-19T17:17:41Z | Public domain (PD-USGov-NSF) | NSF/USAP photo by Steve Clabuesch; uploaded by en:User:Fishdecoy | 1,329,223 | `d978080c729f07abba908fa302533f5c125e89dbdc5a4377c18a5b0b6b8ac561` |

Attribution obligations: CC BY applies to `commons-image-04` and `-09` (CC BY 2.0) and `commons-image-08` (CC BY 4.0). `commons-image-07` is CC0. `commons-image-11` has two parts: a 1945 photograph marked public domain (`PD-Russia-1996`) and a restoration by Adam Cuerden under CC BY 4.0, so it needs attribution too. The maintainer may also want to confirm `PD-Russia-1996` for US use. The other images are public domain as marked (US government works, expired copyright, or `PD-author`).

## Rows (`image-cases.json`)

There is one image-only row per image, plus three rows from `image-cases.json`. These rows add no natural-language text:

- `img-query-prompt` (`commons-image-00`) and `img-document-prompt` (`commons-image-01`): the model's own query and document prompts.
- `mm-two-images`: the text `<|image|> <|image|>` (placeholders only) with `commons-image-02` and `-03`.

There are no text queries, because the text release set covers text. The query-by-image cosine table is therefore empty, unlike the development set, which used agent-written queries.

## Files

| File | SHA-256 | Content |
| --- | --- | --- |
| `image-cases.json` | `8212515810f1e8ad1e707adde4edb8b40c17fdd3352652f7ab2ea9b1326923f1` | the extra rows (`--cases`) |
| `image-embeddings.jsonl` | `13bf5780df73de4e88097bfcb7dcf648d32a789eab1683e2bf12713437769b9d` | per row: rendered ids, per-image sizes and soft-token counts, stage digests, fp32 embedding |
| `image-environment.json` | `3c2a6fd11034df01bc67a8219b1c7b499c1eb3c64608afae72a20b37ee39e8ee` | runner commit, packages, asset and media digests |
| `image-preprocess.json` | `399930720593d96e746b17fea3ddb910fa91161986ded43a6e9a477501d53c0a` | per image: restated preprocessing, PIL vs torchvision backend |
| `image-reference.json` | `25d4febc53633408018e85fca6aa4f9f22e27668cf93952e8a24c0c3aa1ada7f` | reference stability (batch, eager, vision path, load) |
| `images.json` | `888b56cb1e0d49f7b1e94f65eaf4d97e0cba2921d6eea9b763e7f3c97dcb4f29` | media manifest: URL, revision, SHA-1, SHA-256, decoded SHA-256, license, attribution |
| `licenses.json` | `83efc78e2df0a893b4efbfe841a592efcb2b88e6c5a98701b56aa871dcdf8625` | Commons extmetadata and page license templates of all 22 media files, checked 2026-10-14 |
| `tensors.json` | `18c8c1ea1c04d8d69a751ae93403edb5dff09675be16951335e927d508e33ea1` | digests of the per-row stage tensors (not committed) |

`tensors.json` lists the digests of the per-row `.npz` stage tensors, which are large and not committed.

## Reference

`reference/embeddinggemma2/run_embeddinggemma2_image.py --cases image-cases.json` ran at runner commit `b98c9ce` on PyTorch 2.14.1 CPU fp32 (6 threads), sentence-transformers 6.1.0 and transformers 5.19.0. It used `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`, whose assets matched the SHA-256 in the protected development `image-environment.json`. One model process ran at a time on the shared workstation, with no GPU.

```sh
<eg2-video venv>/bin/python model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_image.py --model-dir <snapshot> \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --images <media dir> \
  --cases model-pipeline/proposed/stable/embeddinggemma-2/image-fp32-cpu-release/image-cases.json \
  --out <out> --tensors <tensors>
```

Findings, all inside the development experience:

- 19 rows, no NaN or Inf values.
- 252 to 272 soft tokens per image. Every image is resized. Two images are grayscale (`L`): `commons-image-03` and `-05`.
- The restated preprocessing equals the processor bit for bit on every image.
- Batch of 8 vs single: max abs 7.5e-8.
- Eager vs sdpa attention: 1.4e-7.
- Text-and-vision load vs full load: 0.
- Restated vision path on the soft tokens: 9.5e-7 padded and 1.0e-4 unpadded. The development set gave 1.5e-6 and 8.0e-5.
- PIL backend vs torchvision backend: embedding max abs 1.4e-3, minimum cosine 0.99994.

## Decisions for the reviewer

1. Confirm the 16 licenses above, and the attribution text for the four CC BY images.
2. Accept the set as the P-030 image release set, with the three extra rows.
