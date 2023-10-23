#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <stack>
#include <unordered_map>
#include <vector>

namespace Utilities {
namespace Hacks {
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& arr) {
  for (size_t i = 0; i < std::size(arr); ++i) {
    os << arr[i] << " ";
  }
  return os;
}

template <typename T>
inline std::istream& operator>>(std::istream& in, std::vector<T>& arr) {
  for (size_t i = 0; i < std::size(arr); ++i) {
    in >> arr[i];
  }
  return in;
}
void Faster() {
  std::ios_base::sync_with_stdio(false);
  std::cin.tie(nullptr);
  std::cout.tie(nullptr);
}
}  // namespace Hacks
namespace StingAlgorithms {
namespace AhoCorasick {
const unsigned int kAlpha = 26;
struct Node {
  Node()
      : amount_of_terms_down(0),
        link(std::numeric_limits<unsigned int>::max()),
        dp(std::numeric_limits<unsigned int>::max()),
        compressed(std::numeric_limits<unsigned int>::max()) {
    to.resize(kAlpha + 1, std::numeric_limits<unsigned int>::max());
    go.resize(kAlpha + 1, std::numeric_limits<unsigned int>::max());
    term = false;
  }

  std::vector<unsigned int> to;
  std::vector<unsigned int> go;
  bool term;
  unsigned int amount_of_terms_down;
  unsigned int link;
  unsigned int dp;
  unsigned int compressed;
};

class Trie {
 public:
  Trie() {
    nodes.push_back(Node());
    res.emplace_back(0);
  }

  unsigned int Add(std::string_view str) {
    unsigned int vrtx = 0;
    for (unsigned int i = 0; i < str.size(); ++i) {
      int smb = str[i] - 'a' + 1;
      if (nodes[vrtx].to[smb] == std::numeric_limits<unsigned int>::max()) {
        nodes.push_back(Node());
        nodes[vrtx].to[smb] = nodes.size() - 1;
        res.emplace_back();
        res.back().push_back(i);
      }
      ++nodes[vrtx].amount_of_terms_down;
      vrtx = nodes[vrtx].to[smb];
    }
    nodes[vrtx].term = true;
    ++nodes[vrtx].amount_of_terms_down;
    return vrtx;
  }

  void AhoCorasick() {
    nodes[0].dp = 0;
    nodes[0].compressed = 0;
    for (unsigned int symbol = 1; symbol < kAlpha + 1; ++symbol) {
      nodes[0].go[symbol] =
              (nodes[0].to[symbol] == std::numeric_limits<unsigned int>::max())
                      ? 0
                      : nodes[0].to[symbol];
    }

    std::queue<unsigned int> queue;
    queue.push(0);

    unsigned int from;
    unsigned int to;
    while (!queue.empty()) {
      from = queue.front();
      queue.pop();

      for (unsigned int smb_last = 1; smb_last < kAlpha + 1; ++smb_last) {
        to = nodes[from].to[smb_last];
        auto& vrtx_to = nodes[to];
        if (to == std::numeric_limits<unsigned int>::max()) {
          continue;
        }
        vrtx_to.link = (from != 0) ? nodes[nodes[from].link].go[smb_last] : 0;
        auto& vrtx_link = nodes[vrtx_to.link];
        for (unsigned int smb_new = 1; smb_new < kAlpha + 1; ++smb_new) {
          vrtx_to.go[smb_new] =
                  (vrtx_to.to[smb_new] == std::numeric_limits<unsigned int>::max())
                          ? vrtx_link.go[smb_new]
                          : vrtx_to.to[smb_new];
        }
        vrtx_to.dp = (vrtx_to.term) ? vrtx_link.dp + 1 : vrtx_link.dp;
        vrtx_to.compressed =
                (vrtx_link.term) ? vrtx_to.link : vrtx_link.compressed;
        queue.push(to);
      }
    }
  }
  // private:
  std::vector<Node> nodes;
  std::vector<std::vector<unsigned int>> res;
};
void Solve() {
  /*
   Выведите N строк. В i-й строке выведите несколько чисел: первое — количество
   вхождений Pi в S, далее через пробел выведите индексы вхождение Pi в S в
   возрастающем порядке в 1-индексации.
   */
  std::string str;
  std::cin >> str;
  unsigned int number_of_words;
  std::cin >> number_of_words;
  Trie trie;
  std::vector<unsigned int> num_rtx_in_trie;
  std::vector<int> sz_of_word(number_of_words);
  for (unsigned int i = 0; i < number_of_words; ++i) {
    std::string word;
    std::cin >> word;
    sz_of_word[i] = word.size();
    num_rtx_in_trie.push_back(trie.Add(word) - 1);
  }

  trie.AhoCorasick();

  unsigned int vrtx = 0;
  unsigned int cur_vrtx;
  for (unsigned int i = 0; i < str.size(); ++i) {
    int smb = str[i] - 'a' + 1;
    vrtx = trie.nodes[vrtx].go[smb];
    cur_vrtx = vrtx;

    while (cur_vrtx != 0) {
      if (trie.nodes[cur_vrtx].term) {
        trie.res[cur_vrtx].emplace_back(i);
      }
      cur_vrtx = trie.nodes[cur_vrtx].compressed;
    }
  }
  std::string ans;
  for (unsigned int i = 0; i < number_of_words; ++i) {
    unsigned int size =
            trie.res[num_rtx_in_trie[i] + 1].size();  // размер ответа
    ans += std::to_string(size - 1) + " ";
    for (unsigned int j = 1; j < size; ++j) {
      ans += std::to_string(trie.res[num_rtx_in_trie[i] + 1][j] + 1 -
                            (sz_of_word[i] - 1)) +
             " ";
    }
    ans += '\n';
  }
  std::cout << ans;
}

}  // namespace AhoCorasick
namespace PrefixAndZFunctions {
void ComputePrefixFunction(const std::string_view& ss, std::vector<int>& pi) {
  int nn = ss.size();

  for (int i = 1; i < nn; i++) {
    int jj = pi[i - 1];

    while (jj > 0 && ss[i] != ss[jj]) {
      jj = pi[jj - 1];
    }

    if (ss[i] == ss[jj]) {
      jj++;
    }

    pi[i] = jj;
  }
}
void ZFunction(const std::string_view& str, std::vector<int>& z_func) {
  auto str_sz(static_cast<int>(str.size()));
  int left = -1;
  int right = -1;
  for (int i = 1; i < str_sz; ++i) {
    if (left <= i && i <= right) {
      z_func[i] = std::min(z_func[i - left], right - i + 1);
    }
    while (i + z_func[i] < str_sz && str[z_func[i]] == str[z_func[i] + i]) {
      ++z_func[i];
    }
    if (i + z_func[i] - 1 > right) {
      left = i;
      right = i + z_func[i] - 1;
    }
  }
}
}  // namespace PrefixAndZFunctions
}  // namespace StingAlgorithms
namespace Math {
namespace Log {
static const int kLim = 1000000;
class Logs {
 public:
  Logs() { InitLogs(); };
  void InitLogs() {
    logs_.resize(kLim + 1);
    logs_[1] = 0;
    for (int i = 2; i <= kLim; i++) {
      logs_[i] = logs_[i / 2] + 1;
    }
  }

  size_t GetLogCeil(size_t n) { return logs_[n] + (n % 2 == 0 ? 0 : 1); }

  size_t GetLogFloor(size_t n) { return logs_[n]; }

 private:
  std::vector<size_t> logs_;
};
Logs logs;
}  // namespace Log
}  // namespace Math
namespace Heap {
class Index {
 public:
  Index(unsigned int val, unsigned int aa, unsigned int bb);
  Index(unsigned int val);
  unsigned int x;
  unsigned int pos_min = 0;
  unsigned int pos_max = 0;
};

Index::Index(unsigned int val, unsigned int aa, unsigned int bb)
    : x(val), pos_min(aa), pos_max(bb) {}

Index::Index(unsigned int val) : x(val) { pos_min = pos_max = 0; }

class MinMaxHeap {
 private:
  std::vector<unsigned int> positions_min_heap_{0};
  std::vector<unsigned int> positions_max_heap_{0};
  std::vector<Index> positions_{0};
  unsigned int size_{};

 public:
  MinMaxHeap() = default;
  ~MinMaxHeap() = default;
  void MegaSwapMax(unsigned int i, unsigned int j);
  void MegaSwapMin(unsigned int i, unsigned int j);
  void SiftUpMin(unsigned int v);
  void SiftUpMax(unsigned int v);
  void SiftDownMin(unsigned int v);
  void SiftDownInMaxHeap(unsigned int v);
  unsigned int GetMin();
  unsigned int GetMax();
  void Insert(unsigned int x);
  void SwapIds(unsigned int i, unsigned int j);
  void ExtractMin();
  void ExtractMax();
  [[nodiscard]] unsigned int Size() const;
  void Clear();
};

void MinMaxHeap::MegaSwapMax(unsigned int i, unsigned int j) {
  std::swap(positions_[positions_min_heap_[i]].pos_min,
            positions_[positions_min_heap_[j]].pos_min);
  std::swap(positions_min_heap_[i], positions_min_heap_[j]);
}

void MinMaxHeap::MegaSwapMin(unsigned int i, unsigned int j) {
  std::swap(positions_[positions_max_heap_[i]].pos_max,
            positions_[positions_max_heap_[j]].pos_max);
  std::swap(positions_max_heap_[i], positions_max_heap_[j]);
}

void MinMaxHeap::SiftUpMin(unsigned int v) {
  while (v != 1 and positions_[positions_min_heap_[v]].x <
                            positions_[positions_min_heap_[v / 2]].x) {
    MegaSwapMax(v, v / 2);
    v /= 2;
  }
}

void MinMaxHeap::SiftUpMax(unsigned int v) {
  while (v != 1 and positions_[positions_max_heap_[v]].x >
                            positions_[positions_max_heap_[v / 2]].x) {
    MegaSwapMin(v, v / 2);
    v /= 2;
  }
}

void MinMaxHeap::SiftDownMin(unsigned int v) {
  while (2 * v < positions_min_heap_.size()) {
    unsigned int u = 2 * v;
    if (u + 1 < positions_min_heap_.size() and
        positions_[positions_min_heap_[u]].x >
                positions_[positions_min_heap_[u + 1]].x) {
      ++u;
    }
    if (positions_[positions_min_heap_[v]].x <
        positions_[positions_min_heap_[u]].x) {
      break;
    }
    MegaSwapMax(v, u);
    v = u;
  }
}

void MinMaxHeap::SiftDownInMaxHeap(unsigned int v) {
  while (2 * v < positions_max_heap_.size()) {
    unsigned int u = 2 * v;
    if (u + 1 < positions_max_heap_.size() and
        positions_[positions_max_heap_[u]].x <
                positions_[positions_max_heap_[u + 1]].x) {
      ++u;
    }
    if (positions_[positions_max_heap_[v]].x >
        positions_[positions_max_heap_[u]].x) {
      break;
    }
    MegaSwapMin(v, u);
    v = u;
  }
}

unsigned int MinMaxHeap::GetMin() {
  return positions_[positions_min_heap_[1]].x;
}

unsigned int MinMaxHeap::GetMax() {
  return positions_[positions_max_heap_[1]].x;
}

void MinMaxHeap::Insert(unsigned int x) {
  positions_.emplace_back(x);
  positions_min_heap_.push_back(positions_.size() - 1);
  positions_max_heap_.push_back(positions_.size() - 1);
  positions_.back().pos_min = positions_min_heap_.size() - 1;
  positions_.back().pos_max = positions_max_heap_.size() - 1;
  SiftUpMin(positions_min_heap_.size() - 1);
  SiftUpMax(positions_max_heap_.size() - 1);
  ++size_;
}

void MinMaxHeap::SwapIds(unsigned int i, unsigned int j) {
  std::swap(positions_min_heap_[positions_[i].pos_min],
            positions_min_heap_[positions_[j].pos_min]);
  std::swap(positions_max_heap_[positions_[i].pos_max],
            positions_max_heap_[positions_[j].pos_max]);
  std::swap(positions_[i], positions_[j]);
}

void MinMaxHeap::ExtractMin() {
  MegaSwapMax(1, positions_min_heap_.size() - 1);
  unsigned int v =
          positions_[positions_min_heap_[positions_min_heap_.size() - 1]].pos_max;
  MegaSwapMin(v, positions_max_heap_.size() - 1);
  SwapIds(positions_min_heap_[positions_min_heap_.size() - 1],
          positions_.size() - 1);
  positions_.pop_back();
  positions_min_heap_.pop_back();
  positions_max_heap_.pop_back();
  SiftDownMin(1);
  if (v < positions_max_heap_.size()) {
    if (v != 1 and positions_[positions_max_heap_[v]].x >
                           positions_[positions_max_heap_[v / 2]].x) {
      SiftUpMax(v);
    } else {
      SiftDownInMaxHeap(v);
    }
  }
  --size_;
}

void MinMaxHeap::ExtractMax() {
  MegaSwapMin(1, positions_max_heap_.size() - 1);
  unsigned int v =
          positions_[positions_max_heap_[positions_max_heap_.size() - 1]].pos_min;
  MegaSwapMax(v, positions_min_heap_.size() - 1);
  SwapIds(positions_max_heap_[positions_max_heap_.size() - 1],
          positions_.size() - 1);
  positions_.pop_back();
  positions_min_heap_.pop_back();
  positions_max_heap_.pop_back();
  SiftDownInMaxHeap(1);
  if (v < positions_min_heap_.size()) {
    if (v != 1 and positions_[positions_min_heap_[v]].x <
                           positions_[positions_min_heap_[v / 2]].x) {
      SiftUpMin(v);
    } else {
      SiftDownMin(v);
    }
  }
  --size_;
}

[[nodiscard]] unsigned int MinMaxHeap::Size() const { return size_; }

void MinMaxHeap::Clear() {
  while (size_ > 0) {
    ExtractMin();
  }
}

std::string Request(MinMaxHeap& heap, const std::string& str) {
  if (str == "insert") {
    unsigned int x;
    std::cin >> x;
    heap.Insert(x);
    return "ok\n";
  }
  if (str == "get_min") {
    if (heap.Size() == 0) {
      return "error\n";
    }
    return (std::to_string(heap.GetMin()) + '\n');
  }
  if (str == "extract_min") {
    if (heap.Size() == 0) {
      return "error\n";
    }
    unsigned int ans = heap.GetMin();
    heap.ExtractMin();
    return (std::to_string(ans) + '\n');
  }
  if (str == "get_max") {
    if (heap.Size() == 0) {
      return "error\n";
    }
    return (std::to_string(heap.GetMax()) + '\n');
  }
  if (str == "extract_max") {
    if (heap.Size() == 0) {
      return "error\n";
    }
    unsigned int ans = heap.GetMax();
    heap.ExtractMax();
    return (std::to_string(ans) + '\n');
  }
  if (str == "size") {
    return (std::to_string(heap.Size()) + '\n');
  }
  if (str == "clear") {
    heap.Clear();
    return "ok\n";
  }
  return "";
}
}  // namespace Heap
namespace StructuresForSearchOnSegment {
namespace SegmentTree {
class SegmentTree {
 public:
  SegmentTree(std::vector<int>& arr);
  void Update(int v, int tl, int tr, int index, int x);
  int Find(int v, int tl, int tr, int l, int r);
  int Get(int l, int r) { return Find(1, 0, size_ - 1, l, r); }
  void Set(int index, int x) { Update(1, 0, 42194, index, x); }

 private:
  std::vector<int> arr_;
  int size_;
  void Construction(int v, int tl, int tr, const std::vector<int>& arr);
};

SegmentTree::SegmentTree(std::vector<int>& arr) {
  size_ = static_cast<int>(arr.size());
  arr_.resize(arr.size() * 4, 0);
  Construction(1, 0, size_ - 1, arr);
}

void SegmentTree::Construction(int v, int tl, int tr,
                               const std::vector<int>& arr) {
  if (tl == tr) {
    arr_[v] = arr[tl];
  } else {
    int tm = (tl + tr) / 2;
    Construction(2 * v, tl, tm, arr);
    Construction(2 * v + 1, tm + 1, tr, arr);
    arr_[v] = arr_[2 * v] + arr_[2 * v + 1];
  }
}

void SegmentTree::Update(int v, int tl, int tr, int index, int x) {
  if (tl == tr) {
    arr_[v] = x;
  } else {
    int tm = (tl + tr) / 2;
    if (index > tm) {
      Update(2 * v + 1, tm + 1, tr, index, x);
    } else {
      Update(2 * v, tl, tm, index, x);
    }
    arr_[v] = arr_[2 * v] + arr_[2 * v + 1];
  }
}

int SegmentTree::Find(int v, int tl, int tr, int l, int r) {
  if (l > r) {
    return 0;
  }
  if (tl == l and tr == r) {
    return arr_[v];
  }
  int tm = (tl + tr) / 2;
  int from_left = Find(2 * v, tl, tm, l, std::min(r, tm));
  int from_right = Find(2 * v + 1, tm + 1, tr, std::max(l, tm + 1), r);
  return from_left + from_right;
}
}  // namespace SegmentTree
namespace SparseTable {
template <typename T>
class SparseTable {
 public:
  SparseTable() = default;
  ~SparseTable() = default;
  SparseTable(const std::vector<T>& arr) {
    n_ = arr.size();
    arr_ = arr;
    table_min_.resize(Math::Log::logs.GetLogCeil(n_),
                      std::vector<T>(n_));  // std::bit_width(n_)
    table_max_.resize(Math::Log::logs.GetLogCeil(n_), std::vector<T>(n_));

    for (size_t i = 0; i < n_; i++) {
      table_min_[0][i] = i;  // (arr[i])^-1
      table_max_[0][i] = i;
    }

    for (int kk = 0; (1 << (kk + 1)) <= static_cast<int>(n_) + 1; kk++) {
      for (int i = 0; i + (1 << kk) < static_cast<int>(n_); i++) {
        table_min_[kk + 1][i] =
                std::min(table_min_[kk][i], table_min_[kk][i + (1 << kk)]);
        table_max_[kk + 1][i] =
                std::max(table_max_[kk][i], table_max_[kk][i + (1 << kk)]);
        // table_min_[kk + 1][i] = arr[table_min_[kk][i]] < arr[table_min_[kk][i
        // + (1 << kk)]] ? table_min_[kk][i] : table_min_[kk][i + (1 << kk)];
        // table_max_[kk + 1][i] = arr[table_max_[kk][i]] > arr[table_max_[kk][i
        // + (1 << kk)]] ? table_max_[kk][i] : table_max_[kk][i + (1 << kk)];
      }
    }
  }

  T QueryMin(size_t left, size_t right) {
    ++right;
    size_t kk = Math::Log::logs.GetLogFloor(right - left);
    return std::min(table_min_[kk][left], table_min_[kk][right - (1 << kk)]);
  }

  T QueryMax(size_t left, size_t right) {
    ++right;
    size_t kk = Math::Log::logs.GetLogFloor(right - left);
    return std::max(table_max_[kk][left], table_max_[kk][right - (1 << kk)]);
  }

 private:
  std::vector<T> arr_;
  std::vector<std::vector<T>> table_min_;
  std::vector<std::vector<T>> table_max_;
  size_t n_{};
};
}  // namespace SparseTable
}  // namespace StructuresForSearchOnSegment
namespace SuffixArrayAndLcp {
// BEGIN PRIVATE FOR SuffixArrayAndLcp
namespace Private {
const int kSize = 1048576;
void StableSort(const std::vector<int>& arr, std::vector<int>& cnt) {
  for (int i = 0; i < kSize; i++) {
    cnt[i] = 0;
  }
  int sz = static_cast<int>(arr.size());
  for (int i = 0; i < sz; i++) {
    ++cnt[arr[i]];
  }
  for (int i = 1; i < kSize; i++) {
    cnt[i] += cnt[i - 1];
  }
}

std::vector<int> StringToVector(const std::string_view& ss) {
  std::vector<int> arr(ss.size());
  for (size_t i = 0; i < ss.size(); i++) {
    arr[i] = ss[i];
  }
  return arr;
}
}  // namespace Private
// END PRIVATE FOR SuffixArrayAndLcp

std::vector<int> BuildSuffixArray(const std::string_view& ss) {
  int sz = static_cast<int>(ss.size());
  std::vector<int> classes(sz, -1);  // их не больше sz штук
  std::vector<int> cnt(Private::kSize + sz);
  Private::StableSort(Private::StringToVector(ss), cnt);
  std::vector<int> pp(sz);  // перестановки
  for (int i = sz - 1; i >= 0; i--) {
    pp[--cnt[ss[i]]] = i;
  }
  classes[pp[0]] = 0;
  for (int i = 1; i < sz; i++) {
    classes[pp[i]] = classes[pp[i - 1]];
    if (ss[pp[i]] != ss[pp[i - 1]]) {
      ++classes[pp[i]];
    }
  }
  // теперь отсортируем подстроки длинны kk+1 получается сортировка пар
  // создаём массив пар (classes[i], classes[i+2^kk]) сортируем по второму потом
  // стабильно по первому получаем новый pp и classes
  int kk = 0;
  while (sz - (1 << kk) > 0) {
    std::vector<int> new_pp(sz);
    std::vector<int> new_classes(sz);
    new_pp[0] = (sz + pp[0] - (1 << kk)) % sz;
    for (int i = 1; i < sz; i++) {
      new_pp[i] = (sz + pp[i] - (1 << kk)) % sz;
    }
    Private::StableSort(classes, cnt);
    for (int i = sz - 1; i >= 0; i--) {
      pp[--cnt[classes[new_pp[i]]]] = new_pp[i];
    }  // pp - искомая сортировка для 2^(kk+1)
    new_classes[pp[0]] = 0;
    for (int i = 1; i < sz; i++) {
      new_classes[pp[i]] = new_classes[pp[i - 1]];
      if (classes[pp[i]] != classes[pp[i - 1]] ||
          classes[(pp[i] + (1 << kk)) % sz] !=
                  classes[(pp[i - 1] + (1 << kk)) % sz]) {
        ++new_classes[pp[i]];
      }
    }
    classes = new_classes;
    kk++;
  }
  return pp;
}

std::vector<int> BuildLcpArray(const std::vector<int>& suffix_array,
                               const std::string_view& ss) {
  int sz = static_cast<int>(ss.size());
  std::vector<int> pos(sz);  // массив обратный к suffix_array pp
  for (int i = 0; i < sz; ++i) {
    pos[suffix_array[i]] = i;
  }
  std::vector<int> lcp(sz, 0);
  int kk = 0;
  for (int i = 0; i < sz; ++i) {
    kk = std::max(kk - 1, static_cast<int>(0));
    if (pos[i] == sz - 1) {
      continue;
    }
    int jj = suffix_array[pos[i] + 1];
    while (i + kk < sz && jj + kk < sz && ss[i + kk] == ss[jj + kk]) {
      ++kk;
    }
    lcp[pos[i]] = kk;
  }
  return lcp;
}

void ShowSuffixArray(const std::vector<int>& suffix_array,
                     const std::string_view& ss) {
  for (int i = 0; i < suffix_array.size(); ++i) {
    std::cout << i << ") " << suffix_array[i] << " ";
    for (int j = suffix_array[i]; j < ss.size(); ++j) {
      std::cout << ss[j];
    }
    std::cout << '\n';
  }
}
// FOR LCP
/* class SparseTable {
 public:
  SparseTable(const std::vector<int>& arr) {
    n_ = arr.size();
    table_.resize(std::bit_width(n_), std::vector<int>(n_));

    for (size_t i = 0; i < n_; i++) {
      table_[0][i] = arr[i];
    }

    for (size_t k = 1; (1 << k) <= n_; k++) {
      for (int i = 0; i + (1 << k) <= n_; i++) {
        table_[k][i] = std::min(table_[k - 1][i], table_[k - 1][i + (1 << (k -
1))]);
      }
    }
  }

  int query(size_t left, size_t right) {
    ++right;
    size_t k = std::bit_width(right - left) - 1;
    return std::min(table_[k][left], table_[k][right - (1 << k)]);
  }

 private:
  size_t n_;
  std::vector<std::vector<int>> table_;
};*/
}  // namespace SuffixArrayAndLcp
}  // namespace Utilities


int main() {}
