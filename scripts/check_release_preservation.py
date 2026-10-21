"""Check frozen logos, original demos and memory interfaces without Git/network.

Legacy hashes: GitHub main f48589ec117318a2277b40a72c2b5692560bd018.
Mobile brand hashes: accepted R7 13298e29a5e917189ff74a9c0b9550f752f10192.
The subsequent desktop/gesture UI and README prose may evolve. Original artwork
is default-visible in the linked history page, not frozen into each homepage.
Their existence is not a causal, model-quality or production-readiness claim.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FILE_COUNT = 76
MOBILE_BRAND_COUNT = 113
EXPECTED_SHA256 = {
    "LICENSE": "30c50a4cfcd39f756f0a6970c3ec7edea2bbd24e8e525fb89acf55a511a9254b",
    "claw/__init__.py": "afaefd028c7cc22dd86e74f6b4366690739202713f81c31d39139f98c5682f54",
    "claw/core_memory.py": "5494cc08981094552f05130a4659d43698293e74c04c38bcdddacb4d9f69fcea",
    "claw/memory_manager.py": "a3d91277a322bdac30a1748ed4649c415c70a0da4baa5bc626718103287d1126",
    "claw/models.py": "3befb0e2a256bf3eaf73a59ddd5eaa73be40f8b242523b16c3aecf70696958d1",
    "claw/prompts.py": "1114de51e7054ea62d5ffb7280fead937cbbe4e4141ffe69df7ef89ee6a448b6",
    "claw/semantic_archive.py": "e5ab84279f4cdec956b20e4586aea4f1fb0a9b778a028f6652d385ddbe9bb32a",
    "claw/working_memory.py": "57368266e5034961d8e2f8a45368bf63295a9ade672d897a65cb2ebed46c3488",
    "docs/ECLRR_V4_ARCHITECTURE_IMAGE_PROMPT.md": "b4068187cc3dd4465863d00270fea9ac30136b82ffc4cff3b01f1a0c68da8115",
    "docs/PROJECT_STRUCTURE.md": "20fa0e595fe5f3d1e76dbf25e8c56c67f6ea3efe72c595c57257a15fa054733c",
    "docs/assets/agentic_memory_architecture.png": "4d900d43bb00e723cb5ab0f0c46c7f6ef44820dfff3f21f0f6acd8f94da53241",
    "docs/assets/banner.png": "95b14a3dd028f7ce4bac78de50a92e5cd3b6e7b32f3cdeb70d249d58fef812eb",
    "docs/assets/gpu_knowledge_graph_demo.gif": "a2210b09b40a0dc46da824ec15030ab30dd694a128865bac97b70f8747791557",
    "docs/assets/lifecycle.png": "41a7c721eeb19f9e9f9af98c15432d3185eed08364554d9024ac4cda2b0774ab",
    "docs/assets/memory_kg_observability_demo.gif": "f1f7f7ef5f87c8d6cf68523082edcbc68f231292d2a5a650024851f9ec352707",
    "docs/assets/multi_agent_evolution.png": "6976d0403f14e761407d1956c2e1de94bd87d6eecefe2a82ac8a92b758643642",
    "docs/assets/pipeline.png": "1446ad40f0a7a55cb9655ee7547177cc8869afac0bae5b4ddd4c4067c401e400",
    "docs/assets/sparql_cot.png": "a701eddb095b2f98393e353eabcf153500a38c5adb37f49f74b968bcd5498167",
    "docs/assets/ui_memory_editor_screenshot.png": "672dd69d9615ce373c3de1c81a825b5fb5425a408e4205a31f96fd44b3a63f90",
    "docs/assets/usecase_chat.gif": "b9d7f55f9961f5f4db2d2501d0d35d675de614faf726afd887c7dad3360134e1",
    "docs/assets/usecase_kg.gif": "0ad585564c48a50814301cef5b38cd8297542cd1a589768aa79276a835741b2b",
    "docthinker/memory_core/protocols.py": "accffc79c8a3f6233305e871f01b41545da6ecb8cccd56d8048f71fdbde1e856",
    "docthinker/ui/__init__.py": "a81869533cd9a85d2d7b7a4be9ae54e12f7ec2e75473aa52f2a14d77412b7bd8",
    "docthinker/ui/promo_graph/.gitignore": "823007570221ea8ed87c78a4b85e26f807dd24a1803b06018d8a3614e3305a42",
    "docthinker/ui/promo_graph/__init__.py": "37309372c1cb07811ee292c79913f2bee1f3d411d77ac2f829b145a70baf81d5",
    "docthinker/ui/promo_graph/frontend/e2e/performance.spec.ts": "3f9bfb31a6d88b9958027f49fc9d7b0c27d70332eb1f801c39599e6ae8c0f2f5",
    "docthinker/ui/promo_graph/frontend/e2e/promo-graph.spec.ts": "727803d262ad7f8cedf30f6cf1b1a9e84cb66bf47fb1dfb9c434f6759b89f161",
    "docthinker/ui/promo_graph/frontend/public/mediapipe/hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "docthinker/ui/promo_graph/frontend/src/auto-rotation-controller.ts": "2f8770985e2831adf17d0c8a0485e0fefb168c98ff2a5da615f0ccb1dd137824",
    "docthinker/ui/promo_graph/frontend/src/detail-panel.ts": "82d115be4e2bdd1a4fd5058aea497524fca77d637d2e46363e04a038adb42ef6",
    "docthinker/ui/promo_graph/frontend/src/gesture/mediapipe-adapter.ts": "9b557b69f98d46d61782f508f52a42fa2747ae28a8f9fc8c6aed71f14315efbb",
    "docthinker/ui/promo_graph/frontend/src/gpu-label-layer.ts": "f1ed44f9f003c8b5272d9525e04f1ba01db23004c4033a7b3bb7f23dbe80d9c0",
    "docthinker/ui/promo_graph/frontend/src/graph-data.ts": "9ec18a9bbb0732a8c14a1a3a3126165ba8b92e726938a80b70731808531cea61",
    "docthinker/ui/promo_graph/frontend/src/layout-algorithms.ts": "7cd95a791977b1212a2266799527e15710dc394ebd91904acf76788eb6477efd",
    "docthinker/ui/promo_graph/frontend/src/layout-cache.ts": "ab9efffad753e21ef47fe29bc03a080d7d8927c3e13f0d714c5ab749f1bfb1bd",
    "docthinker/ui/promo_graph/frontend/src/layout.worker.ts": "ba9290dc633898fb57fca4aa4b8d595a57d15172a0f2aa553503080484355bb5",
    "docthinker/ui/promo_graph/frontend/src/semantic-labels.ts": "c14cc12991283302a250fd8de0b78d980d377bc9b247ea068c1878f1f3043035",
    "docthinker/ui/promo_graph/frontend/src/star-map-renderer.ts": "bc84da7c1e01a102c9ee6c7a99fe882eb6005b574e5ff9d3565e34534939a918",
    "docthinker/ui/promo_graph/frontend/src/types.ts": "a0c119eccfe1ced0fd966a4c5537eb1117d4b242b6749627b978925bf7c7b800",
    "docthinker/ui/promo_graph/frontend/src/visual-depth.ts": "e97e2c1f589e6d97e3b65ef8418afa380a955647cd1cd8524f010ca62d13c969",
    "docthinker/ui/promo_graph/frontend/tests/auto-rotation.test.ts": "8001f10ec3eb3a4c4e848eed8d41754bd69f003c2b48e8ac5bded9aa03ce80f3",
    "docthinker/ui/promo_graph/frontend/tests/graph-data.test.ts": "8b3acc3ba28f4961c8a7c28d2bcfd01a878db81d22cfc3507bf1a202dc12fdf5",
    "docthinker/ui/promo_graph/frontend/tests/layout.test.ts": "0d2baea437fcc0c5474f3317f30337d6c2e5c2f0fe1fd411daca022f900b4155",
    "docthinker/ui/promo_graph/frontend/tests/semantic-labels.test.ts": "382f78dddb4e355019a5c30ba37ac6dffe79d42eb1e9077628197444bf317eb3",
    "docthinker/ui/promo_graph/frontend/tests/visual-depth.test.ts": "74da21e748fea0fd94034aeda169af6a99c68afdd2b5a4f2b65b7975fcd1ddc3",
    "docthinker/ui/promo_graph/package-lock.json": "0e576f19b46a2e8ad6a4d3092a3940b5e9417a04d02232bc49736f13919585a8",
    "docthinker/ui/promo_graph/package.json": "497f589a86635b5a6df17ae1fcbdc61083b5b434bf3545f57e4415ad099a5e44",
    "docthinker/ui/promo_graph/playwright.config.ts": "4c654009d2f8237577b0fabdf37a295f55b5d1a04a6ee5e12d7a0131097032d2",
    "docthinker/ui/promo_graph/static/chooser.js": "15838a71fb549ea869ca822426872aa310cc3fb346c6fb2574bacbfa9420cad2",
    "docthinker/ui/promo_graph/static/dist/assets/layout.worker-BDkPaKkA.js": "9479312d0c75d87ef1543ea0f0eada6f7bed5d7cf9f98dab85e46b08c83e7163",
    "docthinker/ui/promo_graph/static/dist/mediapipe/hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "docthinker/ui/promo_graph/static/dist/mediapipe/wasm/vision_wasm_internal.js": "e7fd9858e8e8f221d9b96eddc11f8e077f263e0b7bbd79d3cbe882b134274f8c",
    "docthinker/ui/promo_graph/static/dist/mediapipe/wasm/vision_wasm_internal.wasm": "6a5c64584c2ab61c763b6e204afbdbc7ce1caf7f5216187322bca8df94f646bc",
    "docthinker/ui/promo_graph/static/dist/mediapipe/wasm/vision_wasm_module_internal.js": "1f1d6215324a1fe62f6742d49a3db911170987ca18ad8c1b75f1a1c82acf2b44",
    "docthinker/ui/promo_graph/static/dist/mediapipe/wasm/vision_wasm_module_internal.wasm": "617b8e0248dbd27e9d7ece4218004eae4cefb499196d1bb4fa0e3fef21708756",
    "docthinker/ui/promo_graph/static/dist/mediapipe/wasm/vision_wasm_nosimd_internal.js": "438d1fe8ff7f4d946025bc211c291543c037d8a3785ed4eee60f1f521b236296",
    "docthinker/ui/promo_graph/static/dist/mediapipe/wasm/vision_wasm_nosimd_internal.wasm": "8a3092d34c79d3f57e6ba8592105e8a90f6b07c27891ffecd14cca428bfd3e31",
    "docthinker/ui/promo_graph/static/promo_graph.css": "e5f789005cb4fffa1cd0f5c59bb6c1108d15fef102032d23aac3744fde93c3ae",
    "docthinker/ui/promo_graph/tsconfig.json": "42977753a686c0d767f1c8627a13621979a14cced9a81144339f838d28785036",
    "docthinker/ui/static/logo.png": "c4905f683c23c662b8b2fa5c0a20946845cac5d568364abfc1713acc9170d6e8",
    "docthinker/ui/static/longcatdoc.jpg": "5064b31b2e1ce68c9cef531e4b366af3ef6a9cda844dbdec750b4055ffb11f98",
    "logo.png": "c3b36dd3de1e8ebcb222fcdfc724139ed7b82a16f218763a71bba8e1832b90df",
    "neuro_memory/DESIGN.md": "f6cd2a9c50cb083ea1022f0b7cee225b960fba075e9749713502287bf0c975c2",
    "neuro_memory/VERIFY.md": "cf5dbbf757cec7ce3203d302e4e1ca0ac06f6abf37e6a5d5dab345d4b3c902cc",
    "neuro_memory/__init__.py": "487a6ee9111112d51f41133059fec8ba03f2959eb6edfe4745d21a84bde940f8",
    "neuro_memory/analogical_retrieval.py": "4690e5ad9f9c83714359f255c6ed3d105fcf871687014f69a6a93733f4a39340",
    "neuro_memory/consolidation.py": "6158ebb325b248ccc0f2e6e4d7f928799a86f776e7f7a05d46ce3057b0ff589b",
    "neuro_memory/engine.py": "dab57e4cce09a1b1a0accdfc160c433c061b766206b29c0e2eef2b4d6a33473b",
    "neuro_memory/episode_store.py": "d0a8de7fd824c9017ce9d899bd5544f2eba9e69e38f9f42bbd187c6469c7a9e3",
    "neuro_memory/graph_store.py": "76382d8248fb88f67e2c9a1f4d24c6593a3999ec186c8ea75f46831599fa728a",
    "neuro_memory/models.py": "31055739e3fc65e185df345d782881a968ebca78d60a18254e9245084feb5e60",
    "neuro_memory/spreading_activation.py": "2c0be56e767e5ddb6a6306e7e27aac66ec3aed589d229523bbaed6926ba439b2",
    "scripts/brain_memory_demo.py": "42148c39f86872a5314e0eaad05206de4d640685b26c22a5051b0fe1de189365",
    "scripts/brain_memory_web/app.py": "70c0fc122f99761339e60ec64083b12dde7fd18213b73a2a6a03a0dd25f6389c",
    "scripts/brain_memory_web/templates/index.html": "a29a6bb93f5292ceb3f11a56b7c020c8f74f175a45d4daf7735a72035faa6096",
    "scripts/memory_demo.py": "ff007c7e82bf665dfca77303f80bdbde40a61a45b558065b5c00d4d6ec810eaa",
    "clients/mobile/public/brand/01-evidence-window.svg": "cb5b2a25fe9234e99e2b13f636053d86be0f18c8edfa98c4d4f150392210c0af",
    "clients/mobile/public/brand/02-source-fold.svg": "472e830dc49d418297b73f7bbde2e4bdddb094a91ab7137470fc7d24e9118228",
    "clients/mobile/public/brand/03-reasoning-trail.svg": "392624e8ecc611f5467623e7d1337062f23eaeec3c3b45802ab7093c5429c8de",
    "clients/mobile/public/brand/04-insight-aperture.svg": "e57ed1212a8b1513aea64841825d5b86e6526841fcb1a40701f17f05b253b244",
    "clients/mobile/public/brand/05-knowledge-seed.svg": "3a35ba7256837c7df4f2c6cff1aa4d9e59c2bda715cccb24ba260c49e7c9f096",
    "clients/mobile/public/brand/06-memory-lattice.svg": "0107245b8eda387964d69a826d36b79064d246e75901c62eff1f0e96bfc0221d",
    "clients/mobile/public/brand/catalog.json": "bd253320e4359d1e3a5d195c7d53209c7bdbf76fcd5c43c928a8a5d92d7bc53b",
    "clients/mobile/public/brand/icon-1024.png": "3572ecd1e89ee0eb0646f8b868f4283624b21a9a76efaff44ae20b54ebe9ae31",
    "clients/mobile/public/brand/icon-120.png": "ff093cf9a45810c83b2ef3fdf7fa00ed410780f6dea92dd3f1ce44684c55129d",
    "clients/mobile/public/brand/icon-144.png": "308c2534b5ab6c6f20c6ae1967cbb40370796bb9ea81e1ee1e70a4d6a25ea9ac",
    "clients/mobile/public/brand/icon-152.png": "1ed307d05d3faaf42bc68b9cad7bd69d37ae6e0bd14a2026af4de7f084d19aa0",
    "clients/mobile/public/brand/icon-167.png": "cddc9dacd647a0a68699c9f20aad02bd698107090701c63490e103579086fa80",
    "clients/mobile/public/brand/icon-180.png": "43079b76a6fba679c9b29efa5db7ef8af86f3c4a70dd293ac1c4fcb6fb2a6507",
    "clients/mobile/public/brand/icon-192.png": "def4dbf563c9180748a93c06a31e492ad4b7408f44370e4f0792f6241667399b",
    "clients/mobile/public/brand/icon-20.png": "2d8b3d0d14c98b4156c7cee25a36513073108baba6dee76ae421dc92f77fb2d2",
    "clients/mobile/public/brand/icon-29.png": "c8f7e9539ea57b8a8675509e825ed9b43ea757f61752941abe105b488ea951dc",
    "clients/mobile/public/brand/icon-40.png": "0847177ac45127541df75103c036dcba4f0bcb7d988e5823d6e1c93d1c468eb2",
    "clients/mobile/public/brand/icon-48.png": "8d9a6cea8c5d49ccaeccd0f7205c3e2f4fd0bcece767432c7fb7502c817d7383",
    "clients/mobile/public/brand/icon-512.png": "438f7e82ebd50ce38a99cf2dc8b50f91cd7b00ed60f9304719ebf0e817441f18",
    "clients/mobile/public/brand/icon-58.png": "d6cb9196a4a7e4cfab825783bbb82ae20778b815fbf8f9135c0caa37a51a99f0",
    "clients/mobile/public/brand/icon-60.png": "e528b79421d80e16f2347b67e44ba149c264789365a9f7fa65e04850c39b16b6",
    "clients/mobile/public/brand/icon-72.png": "e41c5313ac074c6d14ab80f6f12ec71d9fc73d477da0abfaaeb2d68f342d4287",
    "clients/mobile/public/brand/icon-76.png": "043275c6d5a6972d5114cea7924c08b36b7c02ceb6710fd878d18f8758d43732",
    "clients/mobile/public/brand/icon-80.png": "8b0b0be72c86c18a23501bb4614f36e8c4424620ecaf92db11739008a92412d8",
    "clients/mobile/public/brand/icon-87.png": "014e8c0402305133ceea8f49d484b567b58a7f93d73b2a5bff6cde8e646dd19c",
    "clients/mobile/public/brand/icon-96.png": "bfc4f642104f238b8ec8a41c4982333c411c99d4d0b3f64c8cff55a8f3a884a8",
    "clients/mobile/public/brand/icon-maskable-512.png": "7a201178f6b81bd2785ac0f56e26edb33d558b6cd27eaf8eb3d100ef2270d0de",
    "clients/mobile/public/brand/icon-maskable.svg": "c59fd566777377882eb041ef60e817182b762fb5f1c7d822b2d83cc36a1e20c4",
    "clients/mobile/public/brand/icon.svg": "aa2ba816f43788cff68a4a708831666b6bfa711ff15be1907f32667f462287e0",
    "clients/mobile/public/brand/index.html": "accccb3f1db8c69ade38a347b4a01305102b415cf6c299d035357d6b71ee7035",
    "clients/mobile/public/brand/logo-16.png": "efe59b3cb989d40827b9dbaa85533b1490d9cabb2bfae01a61d9d23920979d38",
    "clients/mobile/public/brand/logo-256.png": "b4983d70b1f9d65cad753d3d0f45c8784321def0fbcda296f0489fc6203599e1",
    "clients/mobile/public/brand/logo-32.png": "80a89c4b19392a6c7b71093aac44694d81debf99e3ded98a691b97a0b0231772",
    "clients/mobile/public/brand/logo-64.png": "d1850cb156384b9513e2db147ebc76ae82e780def6ffa4d91b6a94d731f2326c",
    "clients/mobile/public/brand/logo-light-64.png": "8d49e8e49bfbb338d22e2ab2f2beee9f7ad61e8780fd02138ab480bf92f45505",
    "clients/mobile/public/brand/logo.svg": "cb5b2a25fe9234e99e2b13f636053d86be0f18c8edfa98c4d4f150392210c0af",
    "clients/mobile/public/brand/showcase.template.html": "6839e53c61350a8be263c25a268f45cf641e936eaa97540b0ba3c2db0918e0a3",
    "clients/mobile/public/brand/small-size-proof.png": "74fd8a6fffe1cd3f7108a127c8e63f813939ba9586487420a856acb5d1a13689",
    "clients/mobile/public/brand/tab-chat-active.png": "2a54c0432127ae4f754025c860d33a270b60d920781d7773042aa92d1088d2aa",
    "clients/mobile/public/brand/tab-chat-active.svg": "e8dca878573c0d3e8bd3221bdcd3201459f098ccc0f3c91eafe5871f7ed7a503",
    "clients/mobile/public/brand/tab-chat.png": "e9d8bc709fd18e0c59cdbdc53929080f7081952da90676d2438d5fa5e2080ec0",
    "clients/mobile/public/brand/tab-chat.svg": "9ceddf423c8aeb14a5f485556b37a980112274267b79f02d54eb105ce18bba1f",
    "clients/mobile/public/brand/tab-documents-active.png": "313ac9328bb0be248f9bf384f24913eff62539a58f11ec1f7b883527cac78b0c",
    "clients/mobile/public/brand/tab-documents-active.svg": "62aaf657f26879e5dc99752fbb3f16b967ea50a7e92038b4a86c6eb333bffec1",
    "clients/mobile/public/brand/tab-documents.png": "c33e39268cf3d40b3ac02190c5f687038afead940b2f8e89abfb0ecae26e9f3c",
    "clients/mobile/public/brand/tab-documents.svg": "d210b82d83456f876b4198ec5ed0ce16f689d57f2d1098e8d46f9a4555ff7080",
    "clients/mobile/public/brand/tab-graph-active.png": "fa7f55993a37c7204d6e2184ef98d5a5bb9ecd118b12f2bd261e4e97c2100a60",
    "clients/mobile/public/brand/tab-graph-active.svg": "8837c1fafc2722561aa43d4f1a8fa1b09180a924f78277f7ab2e6b353213f2a8",
    "clients/mobile/public/brand/tab-graph.png": "e19230b25cc3b384ebd8d125317275f5358cbead75e901b33364287511f154c0",
    "clients/mobile/public/brand/tab-graph.svg": "2314b3322c941818763682074b76b47280979f2913e430a85c99e4eafd3ec4e8",
    "clients/mobile/public/brand/tab-memory-active.png": "1ea4be6b275d7cd3a33522ead37a903eea1b90a1a6552821ed4bddcef4d32655",
    "clients/mobile/public/brand/tab-memory-active.svg": "7a3f899aee9cd811f36b1acdb37b059288548241375722992697dd732fe2d90d",
    "clients/mobile/public/brand/tab-memory.png": "84d4b51e092fae5f54607f9ebee6e80b504b70226f21f5b3c67e874aa1747a05",
    "clients/mobile/public/brand/tab-memory.svg": "438d10cb1d9a19cdaa70161088b44004a16a22c4b91ebeba4a8ee34c30c5c8af",
    "clients/mobile/public/brand/tab-settings-active.png": "3e5271aa01a999f67fed3ed87846d9436e1793b41ab8a19e3e9682ca8f4e8844",
    "clients/mobile/public/brand/tab-settings-active.svg": "fc6649c2d43629eac38e88989528166ca4a190197980223869a5248f885f68e8",
    "clients/mobile/public/brand/tab-settings.png": "aa0ef55564f8b4ac13ac855b5bb5d709a6fc2c17893801651e59dfeb937dc15f",
    "clients/mobile/public/brand/tab-settings.svg": "5f607201cb765c27f974a138a73f99460b7e0c66af7aa15b849323e1bc27b32d",
    "clients/mobile/src/static/brand/01-evidence-window.svg": "cb5b2a25fe9234e99e2b13f636053d86be0f18c8edfa98c4d4f150392210c0af",
    "clients/mobile/src/static/brand/02-source-fold.svg": "472e830dc49d418297b73f7bbde2e4bdddb094a91ab7137470fc7d24e9118228",
    "clients/mobile/src/static/brand/03-reasoning-trail.svg": "392624e8ecc611f5467623e7d1337062f23eaeec3c3b45802ab7093c5429c8de",
    "clients/mobile/src/static/brand/04-insight-aperture.svg": "e57ed1212a8b1513aea64841825d5b86e6526841fcb1a40701f17f05b253b244",
    "clients/mobile/src/static/brand/05-knowledge-seed.svg": "3a35ba7256837c7df4f2c6cff1aa4d9e59c2bda715cccb24ba260c49e7c9f096",
    "clients/mobile/src/static/brand/06-memory-lattice.svg": "0107245b8eda387964d69a826d36b79064d246e75901c62eff1f0e96bfc0221d",
    "clients/mobile/src/static/brand/catalog.json": "bd253320e4359d1e3a5d195c7d53209c7bdbf76fcd5c43c928a8a5d92d7bc53b",
    "clients/mobile/src/static/brand/icon-1024.png": "3572ecd1e89ee0eb0646f8b868f4283624b21a9a76efaff44ae20b54ebe9ae31",
    "clients/mobile/src/static/brand/icon-120.png": "ff093cf9a45810c83b2ef3fdf7fa00ed410780f6dea92dd3f1ce44684c55129d",
    "clients/mobile/src/static/brand/icon-144.png": "308c2534b5ab6c6f20c6ae1967cbb40370796bb9ea81e1ee1e70a4d6a25ea9ac",
    "clients/mobile/src/static/brand/icon-152.png": "1ed307d05d3faaf42bc68b9cad7bd69d37ae6e0bd14a2026af4de7f084d19aa0",
    "clients/mobile/src/static/brand/icon-167.png": "cddc9dacd647a0a68699c9f20aad02bd698107090701c63490e103579086fa80",
    "clients/mobile/src/static/brand/icon-180.png": "43079b76a6fba679c9b29efa5db7ef8af86f3c4a70dd293ac1c4fcb6fb2a6507",
    "clients/mobile/src/static/brand/icon-192.png": "def4dbf563c9180748a93c06a31e492ad4b7408f44370e4f0792f6241667399b",
    "clients/mobile/src/static/brand/icon-20.png": "2d8b3d0d14c98b4156c7cee25a36513073108baba6dee76ae421dc92f77fb2d2",
    "clients/mobile/src/static/brand/icon-29.png": "c8f7e9539ea57b8a8675509e825ed9b43ea757f61752941abe105b488ea951dc",
    "clients/mobile/src/static/brand/icon-40.png": "0847177ac45127541df75103c036dcba4f0bcb7d988e5823d6e1c93d1c468eb2",
    "clients/mobile/src/static/brand/icon-48.png": "8d9a6cea8c5d49ccaeccd0f7205c3e2f4fd0bcece767432c7fb7502c817d7383",
    "clients/mobile/src/static/brand/icon-512.png": "438f7e82ebd50ce38a99cf2dc8b50f91cd7b00ed60f9304719ebf0e817441f18",
    "clients/mobile/src/static/brand/icon-58.png": "d6cb9196a4a7e4cfab825783bbb82ae20778b815fbf8f9135c0caa37a51a99f0",
    "clients/mobile/src/static/brand/icon-60.png": "e528b79421d80e16f2347b67e44ba149c264789365a9f7fa65e04850c39b16b6",
    "clients/mobile/src/static/brand/icon-72.png": "e41c5313ac074c6d14ab80f6f12ec71d9fc73d477da0abfaaeb2d68f342d4287",
    "clients/mobile/src/static/brand/icon-76.png": "043275c6d5a6972d5114cea7924c08b36b7c02ceb6710fd878d18f8758d43732",
    "clients/mobile/src/static/brand/icon-80.png": "8b0b0be72c86c18a23501bb4614f36e8c4424620ecaf92db11739008a92412d8",
    "clients/mobile/src/static/brand/icon-87.png": "014e8c0402305133ceea8f49d484b567b58a7f93d73b2a5bff6cde8e646dd19c",
    "clients/mobile/src/static/brand/icon-96.png": "bfc4f642104f238b8ec8a41c4982333c411c99d4d0b3f64c8cff55a8f3a884a8",
    "clients/mobile/src/static/brand/icon-maskable-512.png": "7a201178f6b81bd2785ac0f56e26edb33d558b6cd27eaf8eb3d100ef2270d0de",
    "clients/mobile/src/static/brand/icon-maskable.svg": "c59fd566777377882eb041ef60e817182b762fb5f1c7d822b2d83cc36a1e20c4",
    "clients/mobile/src/static/brand/icon.svg": "aa2ba816f43788cff68a4a708831666b6bfa711ff15be1907f32667f462287e0",
    "clients/mobile/src/static/brand/logo-16.png": "efe59b3cb989d40827b9dbaa85533b1490d9cabb2bfae01a61d9d23920979d38",
    "clients/mobile/src/static/brand/logo-256.png": "b4983d70b1f9d65cad753d3d0f45c8784321def0fbcda296f0489fc6203599e1",
    "clients/mobile/src/static/brand/logo-32.png": "80a89c4b19392a6c7b71093aac44694d81debf99e3ded98a691b97a0b0231772",
    "clients/mobile/src/static/brand/logo-64.png": "d1850cb156384b9513e2db147ebc76ae82e780def6ffa4d91b6a94d731f2326c",
    "clients/mobile/src/static/brand/logo-light-64.png": "8d49e8e49bfbb338d22e2ab2f2beee9f7ad61e8780fd02138ab480bf92f45505",
    "clients/mobile/src/static/brand/logo.svg": "cb5b2a25fe9234e99e2b13f636053d86be0f18c8edfa98c4d4f150392210c0af",
    "clients/mobile/src/static/brand/tab-chat-active.png": "2a54c0432127ae4f754025c860d33a270b60d920781d7773042aa92d1088d2aa",
    "clients/mobile/src/static/brand/tab-chat-active.svg": "e8dca878573c0d3e8bd3221bdcd3201459f098ccc0f3c91eafe5871f7ed7a503",
    "clients/mobile/src/static/brand/tab-chat.png": "e9d8bc709fd18e0c59cdbdc53929080f7081952da90676d2438d5fa5e2080ec0",
    "clients/mobile/src/static/brand/tab-chat.svg": "9ceddf423c8aeb14a5f485556b37a980112274267b79f02d54eb105ce18bba1f",
    "clients/mobile/src/static/brand/tab-documents-active.png": "313ac9328bb0be248f9bf384f24913eff62539a58f11ec1f7b883527cac78b0c",
    "clients/mobile/src/static/brand/tab-documents-active.svg": "62aaf657f26879e5dc99752fbb3f16b967ea50a7e92038b4a86c6eb333bffec1",
    "clients/mobile/src/static/brand/tab-documents.png": "c33e39268cf3d40b3ac02190c5f687038afead940b2f8e89abfb0ecae26e9f3c",
    "clients/mobile/src/static/brand/tab-documents.svg": "d210b82d83456f876b4198ec5ed0ce16f689d57f2d1098e8d46f9a4555ff7080",
    "clients/mobile/src/static/brand/tab-graph-active.png": "fa7f55993a37c7204d6e2184ef98d5a5bb9ecd118b12f2bd261e4e97c2100a60",
    "clients/mobile/src/static/brand/tab-graph-active.svg": "8837c1fafc2722561aa43d4f1a8fa1b09180a924f78277f7ab2e6b353213f2a8",
    "clients/mobile/src/static/brand/tab-graph.png": "e19230b25cc3b384ebd8d125317275f5358cbead75e901b33364287511f154c0",
    "clients/mobile/src/static/brand/tab-graph.svg": "2314b3322c941818763682074b76b47280979f2913e430a85c99e4eafd3ec4e8",
    "clients/mobile/src/static/brand/tab-memory-active.png": "1ea4be6b275d7cd3a33522ead37a903eea1b90a1a6552821ed4bddcef4d32655",
    "clients/mobile/src/static/brand/tab-memory-active.svg": "7a3f899aee9cd811f36b1acdb37b059288548241375722992697dd732fe2d90d",
    "clients/mobile/src/static/brand/tab-memory.png": "84d4b51e092fae5f54607f9ebee6e80b504b70226f21f5b3c67e874aa1747a05",
    "clients/mobile/src/static/brand/tab-memory.svg": "438d10cb1d9a19cdaa70161088b44004a16a22c4b91ebeba4a8ee34c30c5c8af",
    "clients/mobile/src/static/brand/tab-settings-active.png": "3e5271aa01a999f67fed3ed87846d9436e1793b41ab8a19e3e9682ca8f4e8844",
    "clients/mobile/src/static/brand/tab-settings-active.svg": "fc6649c2d43629eac38e88989528166ca4a190197980223869a5248f885f68e8",
    "clients/mobile/src/static/brand/tab-settings.png": "aa0ef55564f8b4ac13ac855b5bb5d709a6fc2c17893801651e59dfeb937dc15f",
    "clients/mobile/src/static/brand/tab-settings.svg": "5f607201cb765c27f974a138a73f99460b7e0c66af7aa15b849323e1bc27b32d",
}
README_PATHS = ("README.md", "README.zh-CN.md")
README_IMAGES = ("logo.png",)
HISTORY_PATH = "docs/HISTORY.md"
HISTORY_IMAGES = (
    "../logo.png",
    "assets/banner.png",
    "assets/usecase_chat.gif",
    "assets/usecase_kg.gif",
    "assets/memory_kg_observability_demo.gif",
    "assets/gpu_knowledge_graph_demo.gif",
    "assets/agentic_memory_architecture.png",
    "assets/lifecycle.png",
    "assets/multi_agent_evolution.png",
    "assets/pipeline.png",
    "assets/sparql_cot.png",
    "assets/ui_memory_editor_screenshot.png",
)


def _file(root: Path, name: str) -> Path:
    target = root / name
    if target.is_symlink() or not target.is_file() or target.resolve() != target:
        raise ValueError("Protected file missing or symlinked: " + name)
    return target


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _DefaultVisibleImages(HTMLParser):
    """Track disclosure ancestors without treating a closed gallery as shown."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.details_open = []
        self.images = set()
        self.links = set()
        self.visible_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "details":
            # `open` is a boolean HTML attribute, including open="false".
            self.details_open.append(any(name == "open" for name, _ in attrs))
            self.visible_parts.append("\x00")
        elif all(self.details_open):
            self.visible_parts.append(self.get_starttag_text())
            if tag == "img":
                self.images.update(value for name, value in attrs if name == "src" and value)
            if tag == "a":
                self.links.update(value for name, value in attrs if name == "href" and value)

    def handle_startendtag(self, tag, attrs):
        # Details is not a void HTML element: <details/> still opens it.
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag == "details" and self.details_open:
            self.details_open.pop()
            self.visible_parts.append("\x00")
        elif all(self.details_open):
            self.visible_parts.append(f"</{tag}>")

    def handle_data(self, data):
        if all(self.details_open):
            self.visible_parts.append(data)


def _visible_markup(body: str) -> _DefaultVisibleImages:
    # Comments and code examples must not satisfy a real README image contract.
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"(?ms)^\s*(?:```|~~~).*?^\s*(?:```|~~~)[^\n]*$", "", body)
    body = re.sub(r"(`+)[^\n]*?\1", "", body)
    parser = _DefaultVisibleImages()
    parser.feed(body)
    parser.close()
    return parser


def _visible_images(body: str) -> set[str]:
    parser = _visible_markup(body)
    # Keep angle-bracket Markdown destinations intact even though HTMLParser
    # can split them into tag callbacks. A hidden block must not join fragments.
    parser.images.update(re.findall(
        r"!\[[^\]\x00]*\]\(<?([^\s)>\x00]+)>?(?:\s+[^)\x00]*)?\)",
        "".join(parser.visible_parts)))
    return parser.images


def _visible_links(body: str) -> set[str]:
    parser = _visible_markup(body)
    parser.links.update(re.findall(
        r"(?<!!)\[[^\]\x00]+\]\(<?([^\s)>\x00]+)>?(?:\s+[^)\x00]*)?\)",
        "".join(parser.visible_parts)))
    return parser.links


def verify(root: Path = ROOT) -> tuple[int, int]:
    """Verify frozen bytes, the visible archive and bilingual entry points."""
    root = root.resolve(strict=True)
    failures = []
    for name, expected in EXPECTED_SHA256.items():
        try:
            if _digest(_file(root, name)) != expected:
                failures.append(name)
        except (OSError, ValueError):
            failures.append(name)
    for name in README_PATHS:
        try:
            body = _file(root, name).read_text(encoding="utf-8")
            if not set(README_IMAGES).issubset(_visible_images(body)):
                failures.append(name + " (original logo image)")
            if HISTORY_PATH not in _visible_links(body):
                failures.append(name + " (default-visible history link)")
            if "docs/ARCHITECTURE.md" not in _visible_links(body):
                failures.append(name + " (current architecture link)")
        except (OSError, UnicodeError, ValueError):
            failures.append(name)
    try:
        history = _file(root, HISTORY_PATH).read_text(encoding="utf-8")
        if not set(HISTORY_IMAGES).issubset(_visible_images(history)):
            failures.append(HISTORY_PATH + " (original logo/demo/architecture images)")
    except (OSError, UnicodeError, ValueError):
        failures.append(HISTORY_PATH)
    if failures:
        raise ValueError("Preservation check failed: " + ", ".join(failures))
    return len(EXPECTED_SHA256), len(README_PATHS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        count, readmes = verify(args.root)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f"Preserved {count} frozen files ({LEGACY_FILE_COUNT} original + "
          f"{MOBILE_BRAND_COUNT} historical mobile brand); visible history archive "
          f"and {readmes} README original-logo/history entry points intact. "
          "No Git history, network or model calls required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

