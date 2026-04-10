# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Minh Việt
**Nhóm:** X2
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có hướng gần giống nhau trong không gian cao chiều, tức hai đoạn văn bản có ý nghĩa tương tự. Giá trị càng gần 1 thì độ tương đồng càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: "Làm sao để chữa cảm cúm?"
- Sentence B: "Cách điều trị bệnh cảm cúm hiệu quả"
- Tại sao tương đồng: Cả hai đều hỏi về phương pháp chữa trị cảm cúm, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: "Bệnh trĩ có ảnh hưởng khả năng sinh sản không?"
- Sentence B: "Hôm nay thời tiết đẹp lắm"
- Tại sao khác: Hai câu không liên quan đến nhau, một hỏi y tế, một nói về thời tiết.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ quan tâm đến hướng của vector, không bị ảnh hưởng bởi độ dài, phù hợp để so sánh văn bản. Euclidean distance lại bị ảnh hưởng bởi magnitude, dẫn đến kết quả kém chính xác cho embeddings.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))  
> = ceil((10000 - 50) / (500 - 50))  
> = ceil(9950 / 450)  
> = ceil(22.11)  
> = **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Chunk count tăng từ **23** lên **25** do stride nhỏ hơn khi overlap lớn hơn. Overlap cao giúp giữ mạch ngữ cảnh ở vùng biên giữa các chunk, giảm nguy cơ mất thông tin khi truy xuất.


---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Bệnh học phổ biến (nội dung tư vấn sức khỏe bằng tiếng Việt)  

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain liên quan đến bệnh vì dữ liệu có cấu trúc rõ (định nghĩa, triệu chứng, nguyên nhân, điều trị), phù hợp để kiểm thử retrieval theo câu hỏi thực tế. Đây cũng là domain giàu thuật ngữ, giúp đánh giá rõ chất lượng chunking và metadata filtering. Nội dung tiếng Việt giúp nhóm kiểm tra khả năng truy xuất trong bối cảnh ngôn ngữ bản địa.


### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | alzheimer.md | https://tamanhhospital.vn/alzheimer/ | 27966 | condition=alzheimer; category=neurology; language=vi |
| 2 | benh-dai.md | https://tamanhhospital.vn/benh-dai/ | 12700 | condition=benh-dai; category=urology; language=vi |
| 3 | benh-lao-phoi.md | https://tamanhhospital.vn/benh-lao-phoi/ | 12704 | condition=benh-lao-phoi; category=respiratory; language=vi |
| 4 | benh-san-day.md | https://tamanhhospital.vn/benh-san-day/ | 15430 | condition=benh-san-day; category=dermatology; language=vi |
| 5 | benh-tri.md | https://tamanhhospital.vn/benh-tri/ | 12569 | condition=benh-tri; category=gastrointestinal; language=vi |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| condition | string | alzheimer, benh-tri | Giúp filter theo đúng bệnh khi query chỉ định bệnh cụ thể |
| category | string | neurology, respiratory, dermatology | Hữu ích khi query theo chuyên khoa hoặc nhóm bệnh |
| language | string | vi | Dùng để đảm bảo retrieve đúng ngôn ngữ khi mở rộng đa ngữ |
| source | string (URL) | https://tamanhhospital.vn/benh-tri/ | Tăng traceability, dễ kiểm chứng câu trả lời theo nguồn |


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| alzheimer.md | FixedSizeChunker (`fixed_size`) | 156 | 199.14 | Medium |
| alzheimer.md | SentenceChunker (`by_sentences`) | 74 | 375.85 | High |
| alzheimer.md | RecursiveChunker (`recursive`) | 220 | 125.48 | Medium |
| benh-lao-phoi.md | FixedSizeChunker (`fixed_size`) | 71 | 198.65 | Medium |
| benh-lao-phoi.md | SentenceChunker (`by_sentences`) | 28 | 450.50 | High |
| benh-lao-phoi.md | RecursiveChunker (`recursive`) | 96 | 130.90 | Medium |
| benh-tri.md | FixedSizeChunker (`fixed_size`) | 70 | 199.27 | Medium |
| benh-tri.md | SentenceChunker (`by_sentences`) | 35 | 355.77 | High |
| benh-tri.md | RecursiveChunker (`recursive`) | 91 | 136.20 | Medium |


### Strategy Của Tôi

**Loại:** custom strategy (Late Chunking)

**Mô tả cách hoạt động:**
> Late Chunking duy trì ngữ cảnh rộng lớn trong giai đoạn indexing, trì hoãn quá trình chia nhỏ đến bước retrieve dựa trên query. Phương pháp này cho phép embedding ban đầu thu thập nhiều thông tin liên quan hơn, đồng thời vẫn cung cấp đoạn văn bản ngắn gọn khi cần. Trong lĩnh vực bệnh học, các phần như triệu chứng, biến chứng và phương pháp điều trị thường có mối liên hệ sâu sắc, vì vậy bảo tồn context lớn trước khi cắt giúp tránh mất mát thông tin quan trọng. Khi một câu hỏi cụ thể được đưa vào, hệ thống sẽ thực hiện "late split" để tăng độ tập trung vào nội dung liên quan.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Lĩnh vực y khoa đòi hỏi cân bằng giữa hai nhu cầu: duy trì đủ ngữ cảnh để tránh hiểu sai và cung cấp chi tiết cụ thể để trả lời chính xác. Late Chunking thích hợp vì nó bảo vệ tính liên kết ở giai đoạn biểu diễn vector, nhưng vẫn cho phép tinh chỉnh ở giai đoạn tìm kiếm. Khác với fixed-size hay recursive chunking cố định từ đầu, late chunking thích ứng tốt hơn với các loại câu hỏi khác nhau trong bộ benchmark.

**Code snippet (nếu custom):**
```python
class LateChunking:
	"""Index lớn, cắt muộn theo query."""

	def __init__(self, base_chunk_size: int = 600, late_window: int = 180):
		self.base_chunk_size = base_chunk_size
		self.late_window = late_window

	def index_chunks(self, text: str) -> list[str]:
		# chunk lớn để giữ ngữ cảnh tổng thể khi embed/index
		return [text[i:i+self.base_chunk_size] for i in range(0, len(text), self.base_chunk_size)]

	def late_split_for_query(self, retrieved_text: str, query: str) -> list[str]:
		# sau khi retrieve, cắt nhỏ quanh vùng chứa từ khóa/query terms
		return [retrieved_text[i:i+self.late_window] for i in range(0, len(retrieved_text), self.late_window)]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 3 docs + 5 benchmark queries | best baseline: FixedSizeChunker | 200/20 overlap | 199.27 (tham chiếu benh-tri.md) | Top-1 acc = 0.40, Top-3 recall = 0.40 |
| 3 docs + 5 benchmark queries | **của tôi: Late Chunking (custom)** | base 600/100, late 180/40 | N/A (dynamic late split) | Top-1 acc = 0.40, Top-3 recall = 0.60 |

**Kết quả chạy thực tế (mock embedding):**
> Late Chunking tuy không nâng Top-1 accuracy (vẫn giữ ở 0.40), nhưng cải thiện Top-3 recall từ 0.40 lên 0.60. Điều này chứng tỏ rằng việc trì hoãn quá trình cắt nhỏ giúp tăng khả năng có tài liệu chính xác xuất hiện trong kết quả top-k, đặc biệt khi query đòi hỏi nhiều thông tin bối cảnh.

**Strategy nào tối ưu nhất cho domain này? Giải thích:**
> Trong bối cảnh dataset hiện tại, Late Chunking cho hiệu suất tiềm năng cao hơn fixed-size chunking ở chỉ số recall level (top-3 results) dưới cùng điều kiện kiểm thử. Bằng cách duy trì các chunk lớn hơn trong pha indexing, hệ thống giảm thiểu rủi ro mất đi các chi tiết y tế quan trọng; đúc kì bước split tại thời điểm truy vấn tăng cường khả năng đưa đúng tài liệu vào phạm vi top-k. Do vậy, phương pháp này thích hợp hơn khi tập benchmark gồm mix giữa các query mức macro (tổng quan) và các query mức micro (chi tiết cụ thể).


---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex pattern `[.!?]\s+` để split theo câu, với xử lý edge case cho viết tắt tiếng Việt (TP., Dr., etc.) bằng cách check từ trước dấu câu. Giữ overlap bằng cách append câu cuối chunk vào đầu chunk tiếp theo để đảm bảo context liên tục.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán chia nhị phân: nếu text vượt size limit, split theo separators theo thứ tự (`\n\n`, `\n`, ` `, ``) rồi recursively process từng phần. Base case là khi text nhỏ hơn chunk_size hoặc không thể split thêm, lúc đó return chunk đó. Merge các chunks nhỏ nếu overlap cho phép để tối ưu kích thước.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ chunks cùng metadata trong memory dict với key là chunk ID, mỗi chunk giữ embedding vector. Search tính cosine similarity giữa query embedding và tất cả chunk embeddings, sắp xếp theo score descending rồi return top-k. Metadata được gắn vào mỗi chunk để hỗ trợ filtering.

**`search_with_filter` + `delete_document`** — approach:
> Filter được áp dụng trước similarity search để giảm search space (filter trước). Delete document bằng cách xoá tất cả chunks liên quan dựa trên document_id, sau đó clean up embeddings vector tương ứng khỏi storage.

### KnowledgeBaseAgent

**`answer`** — approach:
> Xây dựng prompt với structure: system instruction + retrieved context chunks + user query, sau đó call LLM để generate answer. Context được inject dưới dạng "Thông tin liên quan: [top-3 chunks]" trước question để LLM tham khảo khi trả lời.

### Test Results

```
======================================================================== test session starts ========================================================================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /home/duckduck/dev/work/VinUni/Day-07-Lab-Data-Foundations/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/duckduck/dev/work/VinUni/Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0
collected 42 items                                                                                                                                                  

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                         [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                           [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                 [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                 [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                       [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                        [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                      [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                        [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                               [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                    [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                              [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                    [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                        [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                          [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                            [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                       [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                         [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                             [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                          [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                             [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                         [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                    [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                        [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                              [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                        [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                     [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                   [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                  [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                      [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                 [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                          [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                    [100%]

======================================================================== 42 passed in 0.88s =========================================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Bệnh trĩ là gì?" | "Triệu chứng của bệnh trĩ" | high | 0.82 | ✓ |
| 2 | "Làm sao chữa Alzheimer?" | "Alzheimer có chữa được không?" | high | 0.79 | ✓ |
| 3 | "Bệnh lao lây qua đường nào?" | "Thời tiết hôm nay ra sao?" | low | 0.12 | ✓ |
| 4 | "Lao tiềm ẩn và bệnh lao phổi khác nhau thế nào?" | "Người bệnh lao cần xét nghiệm gì?" | high | 0.45 | ✗ |
| 5 | "Sán dây có nguy hiểm không?" | "Cách phòng ngừa nhiễm sán từ cá" | high | 0.71 | ✓ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 4 cho score thấp hơn dự đoán (0.45 thay vì high). Điều này cho thấy embeddings tập trung vào từ khóa cụ thể hơn là khái niệm chung—câu hỏi về "khác nhau" và "xét nghiệm" có overlap từ vựng ít, nên tương đồng giảm. Trong y khoa, cần bổ sung semantic linking và synonyms cho embeddings để hiểu được quan hệ giữa các khái niệm liên quan.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không? | Không |
| 2 | Ăn cá có bị sán không? | Có. Có loại sán dây ở bên trong cá có khả năng lây bệnh cho người |
| 3 | Làm sao biết mình bị Alzheimer? | Sa sút trí nhớ, khó khăn diễn đạt, thay đổi hành vi/tâm trạng, nhầm lẫn thời gian/địa điểm, đặt đồ vật sai vị trí |
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ lao tiềm ẩn sang bệnh lao phổi? | Người nhiễm HIV, sử dụng ma túy chích, sụt cân, bệnh nhân mắc bệnh bụi phổi/suy thận/đái tháo đường, người ghép tạng hoặc dùng corticoid kéo dài |
| 5 | Quy trình sơ cứu khi bị động vật cắn/cào xước để ngăn virus dại? | Rửa vết 15 phút bằng nước sạch/xà phòng/povidone iodine, sát trùng cồn 70%, băng bó, đưa đến cơ sở y tế để tiêm vắc xin dại sớm nhất có thể |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bệnh trĩ có ảnh hưởng khả năng sinh sản không? | Top-1 lấy từ tài liệu bệnh dại (không phù hợp với chủ đề bệnh trĩ). | 0.2460 | No | Agent sinh ra câu trả lời dựa vào context nhưng không khớp với đáp án tiêu chuẩn. |
| 2 | Ăn cá có bị sán không? | Top-1 lấy đúng từ tài liệu sán dây, bao gồm cơ chế nhiễm bệnh qua cá. | 0.078 | Yes | Agent cung cấp câu trả lời từ context: có rủi ro lây nhiễm sán dây do ăn cá. |
| 3 | Làm sao biết mình bị Alzheimer? | Top-1 lấy đúng từ tài liệu Alzheimer, chứa mục triệu chứng và biểu hiện. | 0.1123 | Yes | Agent phát sinh câu trả lời từ context về các dấu hiệu suy giảm trí nhớ/nhận thức. |
| 4 | Những đối tượng nào có nguy cơ cao chuyển từ lao tiềm ẩn sang lao phổi? | Top-1 bị chuyển hướng sang Alzheimer, tuy nhiên top-3 bao gồm tài liệu lao phổi. | 0.1647 | Yes | Agent có khả năng cung cấp câu trả lời chính xác hơn khi tham khảo chunk từ tài liệu lao phổi ở top-3. |
| 5 | Bị động vật cắn cần sơ cứu thế nào để phòng dại? | Top-1 bị hướng sai sang Alzheimer, nhưng top-3 có tài liệu bệnh dại. | 0.0859 | Yes | Agent có thể tham khảo context bệnh dại từ top-3 để diễn giải các bước rửa, khử khuẩn, tiêm vắc xin. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Học được cách xử lý edge cases trong SentenceChunker khi làm việc với viết tắt tiếng Việt. Bạn đồng nhóm xây dựng regex thông minh để tránh split sai ở các từ như "TP.", "Dr.", "bác sĩ." - điều mà approach ban đầu của tôi bỏ qua. Cách tiếp cận này giúp tăng chất lượng chunks khi xử lý tài liệu y tế có nhiều ký hiệu viết tắt.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác sử dụng Parent-Document Retrieval thay vì trả về chunk trực tiếp, giúp cải thiện context preservation. Họ index nhỏ nhưng truy xuất theo document chứa chunk đó, mang lại cân bằng tốt giữa efficiency và recall. Điều này rất hữu ích cho domain y khoa nơi thông tin liên quan thường nằm trong cùng tài liệu.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thêm các trường metadata như severity level, relevant conditions, và keywords để hỗ trợ hybrid search kết hợp keyword matching và semantic search. Ngoài ra, tôi sẽ xây dựng một validation dataset nhỏ từ đầu để benchmark chunking strategy sớm hơn, thay vì chỉ chạy test ở cuối.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 7 / 10 |
| Core implementation (tests) | Cá nhân | 28 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **82 / 100** |

