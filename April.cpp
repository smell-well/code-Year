#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <tuple>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstring>
#include <iostream>

using namespace std;

struct TrieNode {
    string word;
    unordered_map<char, TrieNode*> children;
    TrieNode() {
        this->word = "";
    }
};

void insertTrie(TrieNode *root, const string &word) {
    TrieNode *node = root;
    for (auto c : word) {
        if (!node->children.count(c)) {
            node->children[c] = new TrieNode();
        }
        node = node->children[c];
    }
    node -> word = word;
}


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class RandomChoice{
public:
    vector<int> sum;
    int n;
    RandomChoice(vector<int>& w) {
        n = w.size();
        sum.push_back(w[0]);
        for (int i = 1; i < n; i++) {
            sum.push_back(sum[i - 1] + w[i]);
            // (*sum)[i] = (*sum)[i - 1] + w[i];
        }
    }
    
    int pickIndex() {
        // cout << n <<endl;
        // return 0;
        int num = rand() % sum[n - 1] + 1;
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (sum[mid] < num) {
                left = mid + 1;
            } else if (sum[mid] >= num) {
                right = mid;
            }
        }
        return left;
    }
};

class SummaryRanges {
private:
    map<int, int> intervals;

public:
    SummaryRanges() {}
    
    void addNum(int val) {
        // 找到 l1 最小的且满足 l1 > val 的区间 interval1 = [l1, r1]
        // 如果不存在这样的区间，interval1 为尾迭代器
        auto interval1 = intervals.upper_bound(val);
        // 找到 l0 最大的且满足 l0 <= val 的区间 interval0 = [l0, r0]
        // 在有序集合中，interval0 就是 interval1 的前一个区间
        // 如果不存在这样的区间，interval0 为尾迭代器
        auto interval0 = (interval1 == intervals.begin() ? intervals.end() : prev(interval1));

        if (interval0 != intervals.end() && interval0->first <= val && val <= interval0->second) {
            // 情况一
            return;
        }
        else {
            bool left_aside = (interval0 != intervals.end() && interval0->second + 1 == val);
            bool right_aside = (interval1 != intervals.end() && interval1->first - 1 == val);
            if (left_aside && right_aside) {
                // 情况四
                int left = interval0->first, right = interval1->second;
                intervals.erase(interval0);
                intervals.erase(interval1);
                intervals.emplace(left, right);
            }
            else if (left_aside) {
                // 情况二
                ++interval0->second;
            }
            else if (right_aside) {
                // 情况三
                int right = interval1->second;
                intervals.erase(interval1);
                intervals.emplace(val, right);
            }
            else {
                // 情况五
                intervals.emplace(val, val);
            }
        }
    }
    
    vector<vector<int>> getIntervals() {
        vector<vector<int>> ans;
        for (const auto& [left, right]: intervals) {
            ans.push_back({left, right});
        }
        return ans;
    }
};

class Solution {
private:
    int tree_min;
    bool flag;
    int count_arr;
    vector<vector<int>> paths;
    int[][] dirs[4][2] = {{0, 1}, {1, 0}, {-1, 0}, {0, -1}};
    int cnt_way;
    vector<string> answer;
    unordered_map<string, int> map;
    int cash;
public:
    int shortestPathLength(vector<vector<int>>& graph) {
        int n = graph.size();
        queue<tuple<int, int, int>> q;
        vector<vector<bool>> map(n, vector<bool>(1 << n));
        int ans = 0;
        for (int i = 0; i < n; i++) {
            q.emplace(i, 1 << i, 0);
            map[i][1 << i] = true;
        }
        while (!q.empty()) {
            auto [index, mask, dist] = q.front();
            // cout << index << " " << mask << " " << dist << endl;
            q.pop();
            if (mask == (1 << n) - 1) {
                // cout << mask << endl;
                ans = dist;
                break;
            } else {
                for (int dest : graph[index]) {
                    // 没有遍历到过
                    int mask_t = mask | (1 << dest);
                    if (!map[dest][mask]) {
                        q.emplace(dest, mask_t, dist + 1);
                        map[dest][mask_t] = true;
                    }
                }
            }
        }
        return ans;
    }

    bool circularArrayLoop(vector<int>& nums) {
        int n = nums.size();
        auto next = [&] (int cur) {
            return ((cur + nums[cur]) % n + n) % n;
        }; 
        for (int  i = 0; i < n; i++) {
            if (!nums[i]) {
                continue;
            }
            int slow = i, fast = next(i);
            while (nums[slow] * nums[fast] > 0 && nums[slow] * nums[next(fast)] > 0) {
                slow = next(slow);
                fast = next(next(fast));              
                if (slow == fast) {
                    if (next(fast) != fast) {
                        return true;
                    }
                    break;
                } 
            }
            int visited = i;
            while (nums[visited] * nums[next(visited)] > 0) {
                int temp = visited;
                visited = next(visited);
                nums[temp] = 0;
            }
        }
        return false;
    }

    void dfs(TreeNode *root, int target) {
        if (root == nullptr) {
            return;
        }
        if (root -> val > target) {
            tree_min = min(tree_min, root -> val);
            flag = true;
        }
        dfs(root -> left, target);
        dfs(root -> right, target);
    }

    int findSecondMinimumValue(TreeNode* root) {
        tree_min = INT32_MAX;
        flag = false;
        // int ans = -1;
        dfs(root, root -> val);
        return flag == true ? -1 : tree_min;
    }

    vector<int> pathInZigZagTree(int label) {
        vector<int> ans;
        // bool odd = false;
        int temp = 1;
        int depth = 1;
        while (temp < label) {
            temp = 2 * temp + 1;
            depth++;
        }
        int num = label;
        while (depth > 0) {
            ans.push_back(label);
            if (depth % 2 == 0) {
                label = (int) (pow(2, depth - 1) * 3 - label - 1);
                label /= 2;
            } else {
                label /= 2;
                label = (int) (pow(2, depth - 2) * 3 - label - 1);
            }
            depth--;
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }

    int titleToNumber(string columnTitle) {
        int ans = 0;
        int n = columnTitle.size();
        int base = 1;
        for (int i = n - 1; i > 0; i--) {
            ans += (columnTitle[i] - 'A' + 1) * base;
            base *= 26;
        }
        return ans;
    }

    void dfs_tree(TreeNode *root, int row, int col, vector<tuple<int, int, int>> &map) {
        if (root == nullptr) {
            return;
        }
        dfs_tree(root -> left, row - 1, col + 1, map);
        map.emplace_back(row, col, root -> val);
        dfs_tree(root -> right, row + 1, col + 1, map);
    }

    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<tuple<int, int, int>> map;
        dfs_tree(root, 0, 0, map);
        vector<vector<int>> ans;
        sort(map.begin(), map.end());
        int temp = std::get<0>(map[0]);
        vector<int> c;
        for (auto& [row, col, value] : map) {
            // cout << row << col <<" "<<value <<endl;
            if (temp == row) {
                c.push_back(value);
                continue;
            }
            ans.push_back(c);
            c = vector<int>();
            c.push_back(value);
            temp = row;
        }
        if (!c.empty()) {
            ans.push_back(c);
        }
        return ans;
    }

    vector<int> kWeakestRows(vector<vector<int>>& mat, int k) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        int r = mat.size(), c = mat[0].size();
        for (int i = 0; i < r; i++) {
            int sum = 0;
            for (int j = 0; j < c; j++) {
                sum += mat[i][j];
            }
            pq.emplace(sum, i);
        }
        vector<int> ans;
        for (int i = 0; i < k; i++) {
            ans.push_back(pq.top().second);
            pq.pop();
        }
        return ans;
    }

    void dijs(vector<vector<int>> &map, vector<int> &dis, vector<bool> &visit, int index) {
        visit[index] = true;
        int n = dis.size() - 1;
        int pis = 0, mindis = INT_MAX;
        // cout << index << endl;
        for (int i = 1; i <= n; i++) {
            if (map[index][i] >= 0 && !visit[i]) {
                dis[i] = min (dis[i], dis[index] + map[index][i]);
            }
        }
        for (int i = 1; i <= n; i++) {
            if (dis[i] < mindis && !visit[i]) {
                mindis = dis[i];
                pis = i;
            }
        }
        if (pis == 0) {
            return;
        }
        dijs(map, dis, visit, pis);
    }
    
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        vector<vector<int>> map(n + 1, vector<int>(n + 1, -1));
        for (auto &it : times) {
            map[it[0]][it[1]] = it[2];
        }
        vector<int> dis(n + 1, INT_MAX);
        vector<bool> visit(n + 1, false);
        dis[k] = 0;
        dijs(map, dis, visit, k);
        int ans = 0;
        for (int i = 1; i <= n; i++) {
            if (dis[i] == INT_MAX) {
                return -1;
            }
            ans = max(dis[i], ans);
        }
        return ans;
    }

    int findUnsortedSubarray(vector<int>& nums) {
        vector<int> arr(nums);
        sort(arr.begin(), arr.end());
        int n = nums.size();
        int left = -1, right = n - 1;
        for (int i = 0; i < n; i++) {
            if (arr[i] == nums[i]) {
                left = i;
                continue;
            }
            break;
        }
        // cout << left << endl;
        for (int i = n - 1; i >= 0; i--) {
            if (arr[i] != nums[i]) {
                right = i;
                break;
            }
        }
        // cout << right << endl;
        return right - left;
    }

    int triangleNumber(vector<int>& nums) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int ans = 0;
        // cout << (nums[0] + nums[1] > nums[3]) << endl;
        for (int i = 0; i < n; i++) {
            // int j = i + 1, c = j + 1;
            if (nums[i] == 0) {
                continue;
            }
            int c = i + 1;
            for (int j = i + 1; j < n; j++) {
                c = max(c, j + 1);
                while (c < n && nums[j] + nums[i] > nums[c]) {
                    c++;
                }
                ans += c - j - 1;
            }
            
        }
        return ans;
    }

    string makeFancyString(string s) {
        string ans = "";
        int cnt = 0;
        int n = s.size();
        ans.push_back(s[0]);
        for (int i = 1; i < n; i++) {
            if (s[i] == s[i - 1]) {
                cnt++;
            } else {
                cnt = 0;
            }
            if (cnt == 2) {
                cnt--;
                continue;
            } else {
                ans.push_back(s[i]);
            }
        }
        return ans;
    }

    bool checkMove(vector<vector<char>>& board, int rMove, int cMove, char color) {
        int r = 8, c = 8;
        // 向上寻找
        for (int i = rMove - 1; i >= 0; i--) {
            if (board[i][cMove] == '.') {
                break;
            }
            if (board[i][cMove] == color) {
                if (abs(rMove - i) >= 2) {
                    // cout << 1<< endl;
                    return true;
                } else {
                    break;
                }
            }
        }

        for (int i = rMove + 1; i < r; i++) {
            if (board[i][cMove] == '.') {
                break;
            }
            if (board[i][cMove] == color) {
                if (abs(rMove - i) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }

        for (int j = cMove - 1; j >= 0; j--) {
            if (board[rMove][j] == '.') {
                break;
            }
            if (board[rMove][j] == color) {
                if (abs(cMove - j) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }

        for (int j = cMove + 1; j < c; j++) {
            if (board[rMove][j] == '.') {
                break;
            }
            if (board[rMove][j] == color) {
                if (abs(cMove - j) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }

        // 斜左向上
        for (int i = 1; (rMove-i) >= 0 && cMove - i >= 0; i++) {
            if (board[rMove - i][cMove - i] == '.') {
                break;
            }
            if (board[rMove - i][cMove - i] == color) {
                if (abs(i) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }

        for (int i = 1; (rMove + i) <r && (cMove + i) < c; i++) {
            if (board[rMove + i][cMove + i] == '.') {
                break;
            }
            if (board[rMove + i][cMove + i] == color) {
                if (abs(i) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }

        // 斜右向上
        for (int j = 1; (rMove - j) >=0 && (cMove + j) < c; j++) {
            if (board[rMove - j][cMove + j] == '.') {
                break;
            }
            if (board[rMove - j][cMove + j] == color) {
                if (abs(j) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }

        for (int j = 1; (rMove + j) < r && (cMove - j) >= 0 ; j++) {
            if (board[rMove + j][cMove - j] == '.') {
                break;
            }
            if (board[rMove + j][cMove - j] == color) {
                if (abs(j) >= 2) {
                    return true;
                } else {
                    break;
                }
            }
        }
        return false;
    }

    int tribonacci(int n) {
        int t[3];
        t[0] = 0; t[1] = 1; t[2] = 1;
        if (n < 3) {
            return t[n];
        }
        int ans = 0;
        for (int i = 3; i <= n; i++) {
            t[i % 3] = t[0] + t[1] + t[2];
            ans = t[i % 3];
        }
        return ans;
    }

    int nthSuperUglyNumber(int n, vector<int>& primes) {
        // sort(primes.begin(), primes.end());
        vector<int> dp(n + 1);
        int m = primes.size();
        unordered_map<int, int> map;
        dp[1] = 1;
        for (int i = 0; i < m; i++) {
            map[i] = 1;
        }
        for (int i = 2; i <= n; i++) {
            int ps = 0, mmin = INT_MAX;
            for (int j = 0; j < m; j++) {
                mmin = min(dp[map[j]] * primes[j], mmin);
            }
            dp[i] = mmin;
            for (int j = 0; j < m; j++) {
                if (mmin == dp[map[j]] * primes[j]) {
                    map[j]++;
                }
            }
        }
        return dp[n];
    }

    int numberOfArithmeticSlices(vector<int>& nums) {
        int n = nums.size();
        if (n < 3) {
            return 0;
        }
        int sub = nums[1] - nums[0];
        int left = 0, ans = 0;
        for (int i = 2; i < n; i++) {
            //cout << sub <<endl;
            if (sub != nums[i] - nums[i - 1]) {
                int len = i - left;
                if (len > 2) {
                    ans += (len - 2) * (len - 1) / 2;
                }
                left = i - 1;
                sub = nums[i] - nums[i - 1];
            }
        }
        if (n - left > 2) {
            int len = n - left;
            ans += (len - 2) * (len - 1) / 2;
        }
        return ans;
    }

    int numberOfArithmeticSlices(vector<int>& nums) {
        int n = nums.size();
        int ans = 0;
        vector<unordered_map<long long, int>> dp(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int d = nums[i] - nums[j];
                int cnt = 0;
                if (dp[j].find(d) == dp[j].end()) {
                    dp[i][d] = 1;
                    continue;
                }
                ans += dp[j][d];
                dp[i][d] += dp[j][d] + 1;
            }
        }
        return ans;
    }

    int longestPalindromeSubseq(string s) {
        int n = s.size();
        vector<vector<int>> dp(n, vector<int>(n));
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s[i] == s[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                } else {
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][n - 1];
    }

    int unhappyFriends(int n, vector<vector<int>>& preferences, vector<vector<int>>& pairs) {
        vector<bool> visited(n);
        unordered_map<int, int> map;
        for (int i = 0; i < n / 2; i++) {
            map[pairs[i][0]] = i;
            map[pairs[i][1]] = i;
        }
        int ans = 0;
        for (int i = 0; i < n / 2; i++) {
            int x = pairs[i][0];
            int y = pairs[i][1];
            for (int j = 0; j < n - 1; j++) {
                if (preferences[x][j] == y) {
                    break;
                }
                // 针对x
                if (!visited[preferences[x][j]]) {
                    int u = preferences[x][j];
                    int v = pairs[map[u]][0] == u ? pairs[map[u]][1] : pairs[map[u]][0];
                    for (int k = 0; k < n - 1; k++) {
                        if (preferences[u][k] == x) {
                            visited[u] = true;
                            ans++;
                            if (!visited[x]) {
                                visited[x] = true;
                                ans++;
                            }
                            break;
                        }
                        if (preferences[u][k] == v) {
                            break;
                        }
                    }
                }
            }
            for (int j = 0; j < n - 1; j++) {
                if (preferences[y][j] == x) {
                    break;
                }
                //针对y
                if (!visited[preferences[y][j]] && preferences[y][j] != x) {
                    int u = preferences[y][j];
                    int v = pairs[map[u]][0] == u ? pairs[map[u]][1] : pairs[map[u]][0];
                    for (int k = 0; k < n - 1; k++) {
                        if (preferences[u][k] == y) {
                            visited[u] = true;
                            ans++;
                            if (!visited[y]) {
                                visited[y] = true;
                                ans++;
                            }
                            break;
                        }
                        if (preferences[u][k] == v) {
                            break;
                        }
                    }
                }
            }
        }
        return ans;
    }

    int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        vector<vector<vector<int>>> dp(m, vector<vector<int>>(n, vector<int>(maxMove)));
        int mod = 1e9 + 7;
        int ans = 0;
        dp[startRow][startColumn][0] = 1;
        for (int k = 1; k < maxMove; k++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (i + 1 < m) {
                        dp[i + 1][j][k] = (dp[i][j][k - 1] + dp[i + 1][j][k]) % mod;
                    } else {
                        ans = (ans + dp[i][j][k - 1]) % mod;
                    }
                    if (i - 1 >= 0) {
                        dp[i - 1][j][k] = (dp[i][j][k - 1] + dp[i - 1][j][k]) % mod;
                    } else {
                        ans = (ans + dp[i][j][k - 1]) % mod;
                    }
                    if (j + 1 < n) {
                        dp[i][j + 1][k] = (dp[i][j][k - 1] + dp[i][j + 1][k]) % mod;
                    } else {
                        ans = (ans + dp[i][j][k - 1]) % mod;
                    }
                    if (j - 1 >= 0) {
                        dp[i][j - 1][k] = (dp[i][j][k - 1] + dp[i][j - 1][k]) % mod;
                    } else {
                        ans = (ans + dp[i][j][k - 1]) % mod;
                    }
                }
            }
        }
        return ans;
    }

    void dfs_arr(int n, int i, vector<bool> &visited) {
        if (i == n + 1) {
            count_arr++;
            return;
        }
        for (int j = 0; j < n; j++) {
            if (!visited[j] && (i % (j + 1) == 0 || (j + 1) % i == 0)) {
                visited[j] = true;
                dfs_arr(n, i + 1, visited);
                visited[j] = false;
            }
        }
    }
    
    int countArrangement(int n) {
        count_arr = 0;
        vector<bool> visited(n);
        dfs_arr(n, 1, visited);
        return count_arr;
    }

    bool checkRecord(string s) {
        int cnt_a = 0;
        int n = s.size();
        int cnt_L = 0;
        for (auto &ch : s) {
            if (ch == 'A') {
                cnt_a++;
            }
            if (ch == 'L') {
                cnt_L++;
                if (cnt_L == 3) {
                    return false;
                }
            } else {
                cnt_L = 0;
            }
        }
        return cnt_a < 2;
    }

    int checkRecord(int n) {
        vector<vector<vector<long>>> dp(n + 1, vector<vector<long>>(3, vector<long>(2)));
        int mod = 1e9 + 7;
        dp[0][0][0] = 0;
        dp[1][1][0] = 1;
        dp[1][0][1] = 1;
        dp[1][0][0] = 1;
        if (n == 1) {
            return 3;
        }
        for (int i = 1; i < n; i++) {
            dp[i + 1][0][0] += ((dp[i][0][0] + dp[i][2][0]) % mod + dp[i][1][0]) % mod;
            dp[i + 1][1][0] += dp[i][0][0];
            dp[i + 1][2][0] += dp[i][1][0];
            dp[i + 1][0][1] += (dp[i][0][0] + dp[i][1][0] + dp[i][2][0] + dp[i][1][1] 
                + dp[i][0][1] + dp[i][2][1]) % mod;
            dp[i + 1][1][1] += dp[i][0][1];
            dp[i + 1][2][1] += dp[i][1][1];            
        }
        int ans = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                ans = (ans + dp[n][i][j]) % mod;
            } 
        }
        return ans;
    }

    void dfs_nums(int i, vector<int> &cur, vector<int> &nums, vector<vector<int>> &ans) {
        int n = nums.size();
        if (i == n) {
            return;
        }
        ans.emplace_back(cur);
        for (int j = i + 1; j < n; j++) {
            cur.push_back(nums[j]);
            dfs_nums(j, cur, nums, ans);
            cur.pop_back();
        }
    }
    
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<int> cur;
        vector<vector<int>> ans;
        dfs_nums(0, cur, nums, ans);
        return ans;
    }

    string reverseVowels(string s) {
        unordered_set<char> set;
        set.emplace('a'); set.emplace('A');
        set.emplace('e'); set.emplace('E');
        set.emplace('i'); set.emplace('I');
        set.emplace('o'); set.emplace('O');
        set.emplace('u'); set.emplace('U');
        int n = s.size();
        int left = 0, right = n - 1;
        while (left < right) {
            while (!set.count(s[left]) && left < right) {
                left++;
            }
            while (!set.count(s[right]) && right > left) {
                right--;
            }
            if (set.count(s[left]) && set.count(s[right])) {
                char temp = s[left];
                s[left] = s[right];
                s[right] = temp;
            }
            left++;
            right--;
        }
        return s;
    }

    string reverseStr(string s, int k) {
        int n = s.size();
        int index = 2 * k;
        while (index < n) {
            reverse(s.begin() + index - 2 * k, s.begin() + index - k + 1);
            index += 2 * k;
        }
        if (index - k < n) {
            reverse(s.begin() + index - 2 * k, s.begin() + index - k + 1);
        } else {
            reverse(s.begin() + index - 2 * k, s.end());
        }
        return s;
    }

    int compress(vector<char>& chars) {
        int ans = 0, n = chars.size();
        int left = 0;
        if (n == 1) {
            return 1;
        }
        for (int i = 0; i <= n - 1; i++) {
            int cnt = 1;
            ans += 1;
            chars[left++] = chars[i];
            while (i < n - 1 && chars[i] == chars[i + 1]) {
                cnt++;
                i++;
            }
            if (cnt > 1) {
                int start = left;
                while (cnt > 0) {
                    ans += 1;
                    chars[left++] = '0' + cnt % 10;
                    cnt /= 10;
                }
                reverse(chars.begin() + start, chars.begin() + left);
            }
        }
        return ans;
    }

    int minTimeToType(string word) {
        int ans = 0;
        char curr = 'a';
        for (auto &ch : word) {
            if (curr != ch) {
                int clockwise = abs(curr - ch);
                int acclock = 26 - clockwise;
                // cout << 'z' - 'b' << endl;
                // cout << min(clockwise, acclock) << endl;
                ans += min(clockwise, acclock);
                curr = ch;
            } 
        }
        return ans + word.size();
    }

    long long maxMatrixSum(vector<vector<int>>& matrix) {
        long long ans = 0;
        int cnt = 0;
        int mmin = INT_MAX;
        for (auto &nums : matrix) {
            for (auto &num : nums) {
                if (num <= 0) {
                    ans += -num;
                    mmin = min(mmin, -num);
                    cnt++;
                } else {
                    ans += num;
                    mmin = min(mmin, num);
                }
                
            }
        }
        //cout << mmin << endl;
        if (cnt % 2 == 1) {
            ans -= 2 * mmin;
        }
        return ans;
    }

    int manhattanDistance(vector<int>& point1, vector<int>& point2) {
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]);
    }

    bool escapeGhosts(vector<vector<int>>& ghosts, vector<int>& target) {
        vector<int> source(2);
        int distance = manhattanDistance(source, target);
        for (auto& ghost : ghosts) {
            int ghostDistance = manhattanDistance(ghost, target);
            if (ghostDistance <= distance) {
                return false;
            }
        }
        return true;
    }

    int getMaximumGenerated(int n) {
        int ans = 0;
        vector<int> nums(101);
        nums[0] = 0;
        nums[1] = 1;
        if (n < 2) {
            return nums[n];
        } else {
            for (int i = 2; i <= n; i++) {
                if (i % 2 == 0) { 
                    nums[i] = nums[i / 2];
                } else {
                    nums[i] = nums[i - 1] + nums[i / 2 + 1];
                }
                ans = max(ans, nums[i]);
            }
        }
        return ans;
    }
    
    int numRescueBoats(vector<int>& people, int limit) {
        int n = people.size();
        int left = 0, right = n - 1;
        int ans = 0;
        sort(people.begin(), people.end());
        while (left <= right) {
            if (people[left] + people[right] <= limit) {
                ans += 1;
                left++;
                right--;
            } else {
                right--;
                ans += 1;
            }
        }
        return ans;
    }

    vector<int> runningSum(vector<int>& nums) {
        int n = nums.size();
        vector<int> ans(n);
        ans[0] = nums[0];
        for (int i = 1; i < n; i++) {
            ans[i] = ans[i - 1] + nums[i];
        }
        return ans;
    }

    int sumOddLengthSubarrays(vector<int>& arr) {
        int n = arr.size();
        long long ans = 0;
        vector<int> num(n + 1);
        for (int i = 0; i < n; i++) {
            num[i + 1] = num[i] + arr[i];
        }
        for (int k = 1; k <= n; k += 2) {
            for (int i = 0; i + k <= n; i++) {
                ans += num[i + k] - num[i];
            }
        }
        return ans;
    }

    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> diff(n + 1);
        for (auto &booking : bookings) {
            diff[booking[0] - 1] += booking[2];
            diff[booking[2]] -= booking[2];
        }
        for (int i = 1; i < n; i++) {
            diff[i] += diff[i - 1];
        }
        diff.pop_back();
        return diff;
    }

    void backtrack(vector<int> &path, int index, vector<vector<bool>> &map) {
        int n = map.size();
        if (index == n - 1) {
            paths.push_back(vector<int>(path));
        }
        for (int i = 0; i < n; i++) {
            if (map[index][i]) {
                path.push_back(i);
                backtrack(path, i, map);
                path.pop_back();
            }
        }
    }
    
    vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
        int n = graph.size();
        vector<vector<bool>> map(n, vector<bool>(n));
        for (int i = 0; i < n; i++) {
            for (auto &node : graph[i]) {
                map[i][node] = true;
            }    
        }
        paths = vector<vector<int>>();
        vector<int> path = vector<int>();
        path.push_back(0);
        backtrack(path, 0, map);
        //paths.push_back(vector<int>(path));
        return paths;
    }
    
    int compareVersion(string version1, string version2) {
        vector<int> ver1, ver2;
        string num;
        for (auto &ch : version1) {
            if (ch != '.') {
                num.push_back(ch);
            } else {
                ver1.push_back(stoi(num));
                num.clear();
            }
        }
        if (!num.empty()) {
            ver1.push_back(stoi(num));
            num.clear();
        }
        for (auto &ch : version2) {
            if (ch != '.') {
                num.push_back(ch);
            } else {
                ver2.push_back(stoi(num));
                num.clear();
            }
        }
        if (!num.empty()) {
            ver2.push_back(stoi(num));
            num.clear();
        }
        int len1 = ver1.size(), len2 = ver2.size();
        int i = 0, j = 0;
        while (i < len1 && j < len2) {
            if (ver1[i] == ver2[j]) {
                i++; j++;
            } else if (ver1[i] > ver2[j]) {
                return 1;
            } else {
                return -1;
            }
        }
        while (i < len1) {
            if (ver1[i] == 0) {
                i++;
            } else {
                return 1;
            }
        }
        while (j < len2) {
            if (ver2[j] == 0) {
                j++;
            } else {
                return -1;
            }
        }
        return 0;
    }

    int search(vector<int>& nums, int target) {
        int n = nums.size();
        int left = 0, right = n - 1;
        int mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] == target) {
                return mid;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    int purchasePlans(vector<int>& nums, int target) {
        int n = nums.size();
        sort(nums.begin(), nums.end());
        int ans = 0, mod = 1e9 + 7;
        int left = 0, right = n - 1;
        while(left < right) {
            while (nums[left] + nums[right] > target && left < right) {
                right--;
            }
            ans = (ans + right - left) % mod;
            left++;
        }
        return ans;
    }

    int balancedStringSplit(string s) {
        int cnt_l = 0, cnt_r = 0;
        int ans = 0;
        for (auto &ch : s) {
            if (ch == 'L') {
                cnt_l++;
            } else {
                cnt_r++;
            }
            if (cnt_r == cnt_l) {
                ans++;
            }
        }
        return ans;
    }

    int findMaximizedCapital(int k, int w, vector<int>& profits, vector<int>& capital) {
        int n = profits.size();
        vector<pair<int, int>> arr;
        for (int i = 0; i < n; i++) {
            arr.push_back({capital[i], profits[i]});
        }
        sort(arr.begin(), arr.end());
        priority_queue<int, vector<int>, less<int>> q;
        int j = 0;
        for (int i = 0; i < k; i++) {
            while (j < n && w >= arr[j].first) {
                q.push(arr[j].second);
                j++;
            }
            if (q.empty()) {
                break;
            } else {
                w += q.top();
                q.pop();
            }
        }
        return w;
    }

    bool check_T(int t, vector<int> &time, int m) {
        int n = time.size();
        int cnt = 0, sum = 0, maxt = 0;
        for (int i = 0; i < n; i++) {
            sum += time[i];
            maxt = max(maxt, time[i]);
            if (sum - maxt > t) {
                cnt++;
                sum = time[i];
                maxt = time[i];
            }
        }
        if (sum != 0) {
            cnt++;
        }
        return cnt <= m;
    }
    
    int minTime(vector<int>& time, int m) {
        int n = time.size();
        int left = 0, right = 1e9;
        int mid = 0;
        // 为什么不能用 <=
        while (left < right) {
            mid = (left + right) / 2;
            if (check_T(mid, time, m)) {
                // 为什么不是 right = mid - 1;
                right = mid;
            } else {
                // 为什么不是 left = mid；
                left = mid + 1;
            }
        }
        return left;
    }
    
    int chalkReplacer(vector<int>& chalk, int k) {
        int sum = 0;
        for (auto &num : chalk) {
            sum += num;
        }
        sum %= k;
        for (int i = 0; i < chalk.size(); i++) {
            sum -= chalk[i];
            if (sum < 0) {
                return i;
            }
        }
        return 0;
    }

    int findIntegers(int n) {
        // 预处理满二叉树时的值
        // 这里使用31是因为后续最开始记数是2，后续还是只用29长度的dp数组
        vector<int> dp(31);
        dp[0] = dp[1] = 1;
        for (int i = 2; i < 31; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        // 处理剩余的位数
        int pre = 0, res = 0;
        for (int i = 29; i >= 0; i++) {
            int val = 1 << i;
            if ((n & val) != 0) {
                // 此时左子树可以直接加入答案
                res += dp[i + 1];
                // 处理右子树
                if (pre == 1) {
                    // 连续两个1
                    break;
                }
                pre = 1;
            } else {
                // 还在左子树
                pre = 0;
            }
            if (i == 0) {
                res++;
            }
        }
        return res;
    }

    int minimumSwitchingTimes(vector<vector<int>>& source, vector<vector<int>>& target) {
        unordered_map<int, int> map;
        int ans = 0;
        for (auto &row : source) {
            for (auto &color : row) {
                if (map.find(color) == map.end()) {
                    map[color] = 1;
                } else {
                    map[color]++;
                }
            }
        }
        for (auto &row : target) {
            for (auto &color : row) {
                if (map.find(color) == map.end() || map[color] == 0) {
                    ans++;
                } else {
                    map[color]--;
                }
            }
        }
        return ans;
    }
    
    int maxmiumScore(vector<int>& cards, int cnt) {
        vector<int> odd, even;
        int n = cards.size();
        for (int i = 0; i < n; i++) {
            if (cards[i] % 2 == 0) {
                even.push_back(cards[i]);
            } else {
                odd.push_back(cards[i]);
            }
        }
        sort(odd.begin(), odd.end(), greater<int>());
        sort(even.begin(), even.end(), greater<int>());
        int ans = 0;
        for (int i = 1; i < odd.size(); i++) {
            odd[i] += odd[i - 1];
            //cout << odd[i] << endl;
        }
        for (int i = 1; i < even.size(); i++) {
            even[i] += even[i - 1];
        }
        // cout << even[0] << endl;
        // 奇数的个数
        for (int i = 0; i <= cnt; i++) {
            int evenCnt = cnt - i;
            // cout << i << endl;
            if (i % 2 != 0) {
                continue;
            }
            if (evenCnt > even.size() || i > odd.size()) {
                continue;
            }
            int sum = 0;
            if (evenCnt == 0) {
                sum = odd[i - 1];
            } else if (i == 0) {
                sum = even[evenCnt - 1];
            } else {
                sum = even[evenCnt - 1] + odd[i - 1];
            }
            // cout << sum << endl;
            ans = max(sum, ans);
        }
        return ans;
    }

    int numberOfBoomerangs(vector<vector<int>>& points) {
        int n = points.size();
        int ans = 0;
        for (int i = 0; i < n; i++) {
            unordered_map<int, int> dist;
            for (int j = 0; j < n; j++) {
                int dis = (points[i][0] - points[j][0]) * (points[i][0] - points[j][0]) + (points[i][1] - points[j][1]) * (points[i][1] - points[j][1]);
                // cout << i << " " << j << " "<< dis << endl;
                // cout << (dist.find(dis) == dist.end()) << endl;
                if (dist.find(dis) == dist.end()) {
                    dist[dis] = 0;
                } else {
                    dist[dis]++;
                    ans += dist[dis] * 2;
                }
            }
        }
        return ans;
    }

    string findLongestWord(string s, vector<string>& dictionary) {
        int n = s.size();
        sort(dictionary.begin(), dictionary.end(), [](const string &a, const string &b) {
            if (a.size() == b.size()) {
                return a < b;
            }
            return a.size() > b.size();
        });
        vector<vector<int>> dp(n + 1, vector<int>(26));
        for (int i = 0; i < 26; i++) {
            dp[n][i] = n;
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = 0; j < 26; j++) {
                if (s[i] - 'a' == j) {
                    dp[i][j] = i;
                } else {
                    dp[i][j] = dp[i + 1][j];
                }
            }
        }
        for (auto &key : dictionary) {
            int index = 0, i = 0;
            while (index < n && i < key.size()) {
                if (dp[index][key[i] - 'a'] == n) {
                    break;
                }
                index = dp[index][key[i] - 'a'] + 1;
                i++;
            }
            if (i == key.size()) {
                return key;
            }
        }
        return "";
    }

    int findPeakElement(vector<int>& nums) {
        int n = nums.size();
        vector<int> arr(n + 2);
        arr[0] = INT_MIN; arr[n + 1] = INT_MIN;
        int left = 1, right = n;
        for (int i = 1; i <= n; i++) {
            arr[i] = nums[i - 1];
        }
        while (left < right) {
            int mid = (left + right) / 2;
            // cout << mid << endl;
            if (arr[mid] > arr[mid - 1] && arr[mid] > arr[mid + 1]) {
                return mid - 1;
            } else if (arr[mid] > arr[mid - 1]) {
                left = mid + 1;
            } else if (arr[mid] > arr[mid + 1]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left - 1;
    }

    void dfs_trie(vector<vector<char>> &board, int x, int y, TrieNode *root, 
        vector<string> &ans) {
        TrieNode *node = root;
        int m = board.size(), n = board[0].size();
        if (!node->children.count(board[x][y])) {
            return;
        } else {
            node = node->children[board[x][y]];
        }
        char prev = board[x][y];
        board[x][y] = '#';
        if (!node->word.empty()) {
            ans.push_back(node->word);
            node->word.clear();
        }
        for (auto dir : dirs) {
            int xx = x + dir[0];
            int yy = y + dir[1];
            if (xx >= 0 && xx < m && yy >= 0 && yy < n && board[xx][yy] != '#') {
                dfs_trie(board, xx, yy, node, ans);
            }
        }
        board[x][y] = prev;
    }

    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        TrieNode root;
        for (auto &word : words) {
            // cout << (&root) << endl;
            insertTrie(&root, word);
        }
        vector<string> ans;
        // cout << CheckAndTakeWord("oath", &root, ans) << endl;
        int m = board.size(), n = board[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dfs_trie(board, i, j, &root, ans);
            }
        }
        return ans;
    }

    bool canWinNim(int n) {
        return n % 4 != 0;
    }

    int minSteps(int n) {
        vector<int> dp(n + 1);
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = i;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                if (i % j == 0) {
                    int times = i / j;
                    dp[i] = min(dp[i], dp[j] + times);
                }
            }
        }
        return dp[n];
    }

    int lengthOfLastWord(string s) {
        int n = s.size();
        int i = n - 1;
        while (i >= 0) {
            if (s[i] == ' ') {
                i--;
                continue;
            }
            // 不为空格
            int end = i;
            while (i >= 0 && s[i] != ' ') {
                i--;
            }
            return end - i;
        }
        return 0;
    }

    vector<ListNode*> splitListToParts(ListNode* head, int k) {
        ListNode *temp = head;
        int all = 0;
        while (temp != nullptr) {
            all++;
            temp = temp->next;
        }
        vector<ListNode *> answer;
        int num = all / k, mod = all % k;
        temp = head;
        ListNode *listhead = head;
        int cnt = 0;
        while (temp != nullptr) {
            cnt++;
            // 达到均分的最低限
            if (mod != 0) {
                if (cnt == num + 1) {
                    mod--;
                    ListNode *prev = temp;
                    temp = temp->next;
                    prev->next = nullptr;
                    answer.push_back(listhead);
                    listhead = temp;
                    cnt = 0;
                    continue;
                }
            } else {
                if (cnt == num) {
                    ListNode *prev = temp;
                    temp = temp->next;
                    prev->next = nullptr;
                    answer.push_back(listhead);
                    listhead = temp;
                    cnt = 0;
                    continue;
                }
            }
            temp = temp->next;
        }
        while (answer.size() != k) {
            if (temp == nullptr) {
                answer.push_back(nullptr);
            } else {
                answer.push_back(temp);
                temp = temp->next;
            }
        }
        return answer;
    }

    bool isPowerOfThree(int n) {
        while (n > 1) {
            if (n % 3 != 0) {
                return false;
            }
            n /= 3;
        }
        return n == 1;
    }

    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        if (n == 0 || m == 0) {
            return abs(n - m);
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        // cout << dp[n - 1][m - 1] <<endl;
        int ans = n + m - 2 * dp[n][m];
        return ans;
    }

    int getSum(int a, int b) {
        int ans = 0;
        int carry = 0;
        for (int i = 0; i < 32; i++) {
            int a_base = (a >> i) & 1, b_base = (b >> i) & 1;
            // cout << (a_base ^ b_base) << " " << carry << endl;
            // 两个位不相同 01
            if ((a_base ^ b_base)) {
                if (carry == 0) {
                    ans |= 1 << i;
                }
            } else { // 两个位相同 00 11
                if (a_base) {
                    if (carry != 0) {
                        ans |= 1 << i;
                    } else {
                        carry = 1;
                    }
                } else {
                    if (carry != 0) {
                        ans |= 1 << i;
                        carry = 0;
                    }
                }
            }
        }
        return ans;
    }

    int numDecodings(string s) {
        int n = s.size();
        int mod = 1e9 + 7;
        vector<long long> dp(n + 1);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            // 单独一位无论如何都可以表示一个字母(除零外)
            if (s[i - 1] == '*') {
                dp[i] = ((9 * dp[i - 1]) % mod + dp[i]) % mod;
            } else if (s[i - 1] != '0') {
                dp[i] = (dp[i - 1] + dp[i]) % mod;
            }
            // 两个字母要满足小于等于26的条件
            if (i > 1 && ((s[i - 2] >= '1' && s[i - 2] <= '2') || s[i - 2] == '*')) {
                if (s[i - 2] == '*') {
                    // * = 1
                    if (s[i - 1] == '*') {
                        dp[i] =  ((9 * dp[i - 2]) % mod + dp[i]) % mod;
                    } else {
                        dp[i] = (dp[i - 2] + dp[i]) % mod;
                    }
                    // * = 2;
                    if (s[i - 1] <= '6' && s[i - 1] >= '0') {
                        dp[i] = (dp[i - 2] + dp[i]) % mod;
                    } else if (s[i - 1] == '*') {
                        dp[i] = ((6 * dp[i - 2]) % mod + dp[i]) % mod;
                    }
                } else {
                    if (s[i - 2] == '2') {
                        if (s[i - 1] <= '6' && s[i - 1] >= '0') {
                            dp[i] = (dp[i - 2] + dp[i]) % mod;
                        }
                        if (s[i - 1] == '*') {
                            dp[i] = ((6 * dp[i - 2]) % mod + dp[i]) % mod;
                        }
                    } else {
                        dp[i] = (dp[i - 2] + dp[i]) % mod;
                        if (s[i - 1] == '*') {
                            dp[i] = ((8 * dp[i - 2]) % mod + dp[i]) % mod;
                        }
                    }  
                }
                
            }
        }
        return dp[n];
    }

    void dfs_way(stack<int> &sum, TreeNode *root, int target) {
        if (root == nullptr) {
            return;
        }
        sum.push(root->val);
        stack<int> temp = sum;
        long long total = 0;
        int n = temp.size();
        for (int i = 0; i < n; i++) {
            total += temp.top();
            if (total == target) {
                // cout << root->val << endl;
                cnt_way++;
            }
            temp.pop();
        }
        dfs_way(sum, root->left, target);
        dfs_way(sum, root->right, target);
        sum.pop();
    }

    int pathSum(TreeNode* root, int targetSum) {
        cnt_way = 0;
        stack<int> stk;
        dfs_way(stk, root, targetSum);
        return cnt_way;
    }

    int findMinMoves(vector<int>& machines) {
        int n = machines.size();
        int sum = 0;
        for (auto &num : machines) {
            sum += num;
        }
        if (sum % n != 0) {
            return -1;
        }
        int avg = sum / n;
        int ans = 0, total = 0;
        for (auto num : machines) {
            num -= avg;
            // 把左边看成一个整体
            total += num;
            ans = max(ans, max(abs(num), total));
        }
        return ans;
    }

    int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
        int s1 = (ax2 - ax1) * (ay2 - ay1);
        int s2 = (bx2 - bx1) * (by2 - by1);
        int L = min(ax2, bx2) - max(ax1, bx1);
        int W = min(by2, ay2) - max(ay1, by1);
        if (L > 0 && W > 0 ) {
            // 防止不重叠的时候
            return s1 + s2 - L * W;
        }
        return s1 + s2;
    }

    string toHex(int num) {
        auto trans = [](int a) {
            if (a < 10) {
                return '0' + a;
            } else {
                return 'a' + a - 10;
            }
        };
        int mask = 15;
        string ans = "";
        if (num == 0) {
            return "0";
        }
        for (int i = 0; i <= 28; i += 4) {
            int dig = (num >> i) & mask;
            cout << dig << endl;
            ans.push_back(trans(dig));
        }
        // 排除前导零
        while (ans[ans.size() - 1] == '0') {
            ans.pop_back();
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }

    int peakIndexInMountainArray(vector<int>& arr) {
        int n = arr.size();
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (arr[mid] > arr[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        for (auto &row : matrix) {
            for (auto &num : row) {
                if (target == num) {
                    return true;
                }
            }
        }
        return false;
    }

    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m = nums2.size();
        stack<int> stk;
        unordered_map<int, int> map;
        for (int i = 0; i < m; i++) {
            while (!stk.empty() && stk.top() < nums2[i]) {
                map[stk.top()] = nums2[i];
                stk.pop();
            }
            stk.push(nums2[i]);
        }
        while (!stk.empty()) {
            map[stk.top()] = -1;
            stk.pop();
        }
        for (int i = 0; i < n; i++) {
            nums1[i] = map[nums1[i]];
        }
        return nums1;
    }

    void dfs_str(string s, stack<char> &stk, string ans, int index, int cnt) {
        int n = s.size();
        // cout << ans << endl;
        if (cnt > n - index) {
            return;
        }
        for (int i = index + 1; i < n; i++) {
            if (s[i] != '(' && s[i] != ')') {
                ans.push_back(s[i]);
            } else {
                // 先确定这个括号删不删除
                // 删除
                if (cnt > 0) {
                    stack<char> next = stk;
                    dfs_str(s, next, ans, i, cnt - 1);
                }
                // 不删除
                if (stk.empty()) {
                    stk.push(s[i]);
                } else {
                    if (stk.top() == ')') {
                        break;
                    } else {
                        if (s[i] == ')') {
                            stk.pop();
                        } else {
                            stk.push(s[i]);
                        }
                    }
                }
                ans.push_back(s[i]);
                // dfs_str(s, stk, ans, i, cnt);
            }
        }
        if (cnt == 0 && stk.empty()) {
            if (map.find(ans) == map.end()) {
                map[ans] = 1;
                answer.push_back(ans);
            }
            return;
        }
    }
    
    vector<string> removeInvalidParentheses(string s) {
        stack<char> stk;
        for (auto &ch : s) {
            if (ch != '(' && ch != ')') {
                continue;
            } else {
                if (stk.empty()) {
                    stk.push(ch);
                } else {
                    if (stk.top() == '(' && ch == ')') {
                        stk.pop();
                    } else {
                        stk.push(ch);
                    }
                }
            }
        }
        int n = stk.size();
        stack<char> sk;
        answer = vector<string>();
        map = unordered_map<string, int>();
        string temp = "";
        dfs_str(s, sk, temp, -1, n);
        return answer;
    }

    bool reorderedPowerOf2(int n) {
        unordered_map<int, vector<int>> map;
        int base = 1;
        while (base <= 1e9) {
            vector<int> count(10);
            int temp = base;
            while (temp != 0) {
                count[temp % 10]++;
                temp /= 10;
            }
            map[base] = count;
            base *= 2;
        }
        vector<int> target(10);
        while (n != 0) {
            target[n % 10]++;
            n /= 10;
        }
        // cout << target[1] << endl;
        for (auto &[key, value] : map) {
            bool flag = true;
            for (int i = 0; i < 10; i++) {
                if (value[i] != target[i]) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                return true;
            }
        }
        return false;
    }

    vector<string> findWords(vector<string>& words) {
        vector<unordered_set<char>> map(3);
        vector<string> str(3);
        str[0] = "qwertyuiop";
        str[1] = "asdfghjkl";
        str[2] = "zxcvbnm";
        for (int i = 0; i < 3; i++) {
            for (auto &ch : str[i]) {
                map[i].insert(ch);
            }
        }
        vector<string> ans;
        for (auto &word : words) {
            for (int i = 0; i < 3; i++) {
                bool flag = true;
                for (auto ch : word) {
                    if (ch >= 'A' && ch <= 'Z') {
                        ch += 'a' - 'A';
                    }
                    if (map[i].count(ch) == 0) {
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    ans.push_back(word);
                    continue;
                }
            }
        }
        return ans;
    }

    int distributeCandies(vector<int>& candyType) {
        int n = candyType.size();
        unordered_map<int, int> map;
        for (auto &candy : candyType) {
            map[candy]++;
        }
        if (map.size() >= n / 2) {
            return n / 2;
        } else {
            return map.size();
        }
    }

    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        ListNode *child = node->next;
        node->next = child->next;
    }

    int longestSubsequence(vector<int>& arr, int difference) {
        int n = arr.size();
        int ans = 0;
        unordered_map<int, int> map;
        for (int i = 0; i < n; i++) {
            int temp = arr[i] - difference;
            if (map.find(temp) == map.end()) {
                map[arr[i]] = 1;
            } else {
                map[arr[i]] = map[temp] + 1;
            }
            ans = max(ans, map[arr[i]]);
        }
        return ans;
    }

    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        vector<bool> arr(n + 1);
        for (int i = 0; i < n; i++) {
            arr[nums[i]] = true;
        }
        for (int i = 0; i <= n; i++) {
            if (!arr[i]) {
                return i;
            }
        }
        return 0;
    }

    string getHint(string secret, string guess) {
        unordered_map<char, int> sec, gue;
        int n = secret.size();
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (secret[i] == guess[i]) {
                cnt++;
            } else {
                sec[secret[i]]++;
                gue[guess[i]]++;
            }
        }
        int all = 0;
        for (auto &[key, value] : gue) {
            all += min(value, sec[key]);
        }
        string ans;
        ans.push_back('0' + cnt);
        ans.push_back('A');
        ans.push_back('0' + all);
        ans.push_back('B');
        return ans;
    }

    vector<int> addNegabinary(vector<int>& arr1, vector<int>& arr2) {
        reverse(arr1.begin(), arr1.end());
        reverse(arr2.begin(), arr2.end());
        int n = arr1.size(), m = arr2.size();
        int carry = 0;
        vector<int> ans;
        for (int i = 0; i < min(m, n); i++) {
            int temp = arr1[i] + arr2[i] + carry;
            if (temp >= 2) {
                temp -= 2;
                carry = -1;
            } else if (temp == -1) {
                temp = 1;
                carry = 1;
            } else {
                carry = 0;
            }
            ans.push_back(temp);
        }
        if (m < n) {
            for (int i = m; i < n; i++) {
                int temp = arr1[i] + carry;
                if (temp >= 2) {
                    temp -= 2;
                    carry = -1;
                } else if (temp == -1) {
                    temp = 1;
                    carry = 1;
                } else {
                    carry = 0;
                }
                ans.push_back(temp);
            }
        } else {
            for (int i = n; i < m; i++) {
                int temp = arr2[i] + carry;
                if (temp >= 2) {
                    temp -= 2;
                    carry = -1;
                } else if (temp == -1) {
                    temp = 1;
                    carry = 1;
                } else {
                    carry = 0;
                }
                ans.push_back(temp);
            }
        }
        if (carry == -1) {
            ans.push_back(1);
            ans.push_back(1);
        } else if (carry == 1) {
            ans.push_back(1);
        }
        // 去除前导零
        while (ans.size() > 1 && ans[ans.size() - 1] == 0) {
            ans.pop_back();
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }

    int findPoisonedDuration(vector<int>& timeSeries, int duration) {
        int total = 0;
        int n = timeSeries.size();
        int start = timeSeries[0];
        for (int i = 1; i < n; i++) {
            if (timeSeries[i] >= start + duration) {
                total += duration;
            } else {
                total += timeSeries[i] - start;
            }
            start = timeSeries[i];
        }
        total += duration;
        return total;
    }

    int kInversePairs(int n, int k) {
        int MOD = 1e9 + 7;
        vector<vector<int>> dp(n + 1, vector<int>(1001));
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= k; j++) {
                dp[i][j] = (j - 1 >= 0 ? dp[i][j - 1] : 0) - (j - i >= 0 ? dp[i - 1][j - i] : 0) + dp[i - 1][j];
                if (dp[i][j] > MOD) {
                    dp[i][j] -= MOD;
                } else if (dp[i][j] < 0) {
                    dp[i][j] += MOD;
                }
            }
        }
        return dp[n][k];
    }

    int getMoneyAmount(int n) {
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
        for (int i = n; i > 0; i--) {
            for (int j = i + 1; j <= n; j++) {
                int temp = INT_MAX;
                for (int k = i; k < j; k++) {
                    temp = min(k + max(dp[i][k - 1], dp[k + 1][j]), temp);
                }
                dp[i][j] = temp;
            }
        }
        return dp[1][n];
    }

    bool detectCapitalUse(string word) {
        int pis = -1, upper = 0;
        int n = word.size();
        for (int i = 0; i < n; i++) {
            if (word >= 'A' && word <= 'Z') {
                pis = i;
                upper++;
            }
        }
        return (upper == n || pis == 0 || upper == 0);
    }


};



