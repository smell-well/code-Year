#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string.h>
#include <math.h>
#include <algorithm>

using namespace std;

#define null NULL

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

struct Edge {
    int len, x, y;
    Edge(int len, int x, int y) : len(len), x(x), y(y) {}
};

class Solution {
public:
    int target;
    bool flag;
    int minCostClimbingStairs(vector<int>& cost) {
        int ans = 0;
        int step0 = cost[0];
        int step1 = cost[1];
        int len = cost.size();
        if (len < 3) {
            return min(step1, step0);
        }
        for(int i=2; i<len; i++) {
            ans = min(step0, step1) + cost[i];
            step0 = step1;
            step1 = ans;
        }
        ans = min(step0, step1);
        return ans;
    }
    
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans = vector<int>();
        priority_queue<pair<int, int>> pq;
        for(int i=0; i<k; i++) {
            pq.emplace(nums[i], i);
        }
        ans.push_back(pq.top().first);
        int n = nums.size();
        for(int i=k; i<n; i++) {
            pq.emplace(nums[i], i);
            while(pq.top().second <= i - k) {
                pq.pop();
            }
            ans.push_back(pq.top().first);
        }
        return ans;
    }

    vector<int> maxSlidingWindow_monotonic(vector<int>& nums, int k) {
        vector<int> ans = vector<int>();
        deque<int> q = deque<int>();
        for(int i=0; i<k; i++) {
            while(!q.empty() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
        }
        ans.push_back(nums[q.front()]);
        int n = nums.size();
        for(int i=k; i<n; i++) {
            while(!q.empty() && nums[i] >= nums[q.back()]) {
                q.pop_back();
            }
            q.push_back(i);
            while(q.front() <= i - k) {
                q.pop_front();
            }
            ans.push_back(nums[q.front()]);
        }
        return ans;
    }

    vector<int> maxSlidingWindow_fix(vector<int>& nums, int k) {
        vector<int> ans = vector<int>();
        int n = nums.size();
        vector<int> prefix(n), suffix(n);
        prefix[0] = nums[0];
        for(int  i=1; i<n; i++) {
            if(i % k == 0) {
                prefix[i] = nums[i];
            } else {
                prefix[i] = max(nums[i], prefix[i-1]);
            }
        }

        suffix[n - 1] = nums[n - 1];
        for(int i = n-2; i >= 0; i--) {
            if ((i + 1) % k == 0) {
                suffix[i] = nums[i];
            } else {
                suffix[i] = max(nums[i], suffix[i + 1]);
            }
        }
        
        for(int i = 0; i < n - k + 1; i++) {
            ans.push_back(max(suffix[i], prefix[i + k - 1]));
        }
        return ans;
    }

    ListNode* partition(ListNode* head, int x) {
        ListNode *pos = head, *prev = head;
        ListNode *last = NULL;
        while(pos != null) {
            if (pos -> val < x) {
                if (last == null) {
                    prev->next = pos->next;
                    if (pos != head) {
                        pos->next = head;
                        head = pos;
                    }
                } else {
                    if (last->next == pos) {
                        prev = pos;
                    } else {
                        prev->next = pos->next;
                        pos->next = last->next;
                        last->next = pos;
                    }
                }
                last = pos;
                pos = prev;
            }
            prev = pos;
            pos = pos->next;
        }
        return head;
    }

    vector<vector<int>> largeGroupPositions(string s) {
        vector<vector<int>> ans = vector<vector<int>>();
        int start = 0, end = 0;
        int n = s.length();
        while(end != n) {
            if (start == end || s[start] == s[end]) {
                end++;
            } else {
                if (end - start >= 3) {
                    vector<int> node = vector<int>();
                    node.push_back(start);
                    node.push_back(end - 1);
                    ans.push_back(node);
                }
                start = end;
                end++;
            }
        }

        // 还需要考虑end到头时的情况
        if (end - start >= 3) {
            vector<int> node = vector<int>();
            node.push_back(start);
            node.push_back(end - 1);
            ans.push_back(node);
        }
        return  ans;
    }

    bool isValidBST(TreeNode* root) {
        flag = false;
        target = 0;
        dfs_BST(root);
    }

    bool dfs_BST(TreeNode* root) {
        if (root == null) {
            return true;
        }
        if (dfs_BST(root->left)) {
            //已找到数的最小值
            if(flag) {
                if (root->val <= target) {
                    return false;
                }
            } else {
                target = root->val;
            }
            return dfs_BST(root->right);

        }
        return false;
    }

    vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
        int n = values.size();
        int cnt = 0;
        unordered_map<string, int> variables;
        for (int i = 0; i < n; i++) {
            if (variables.find(equations[i][0]) == variables.end()) {
                variables[equations[i][0]] = cnt++;
            }
            if (variables.find(equations[i][1]) == variables.end()) {
                variables[equations[i][1]] = cnt++;
            }
        }
        vector<int> father(cnt);
        vector<double> weight(cnt, 1.0);
        for (int i = 0; i < cnt; i++) {
            father[i] = i;
        }

        for (int i = 0; i < n; i++) {
            int a = variables[equations[i][0]], b = variables[equations[i][1]];
            Union(father, weight, b, a, values[i]);
        }

        int qn = queries.size();
        vector<double> ans(qn);
        for (int i = 0; i < qn; i++) {
            string a = queries[i][0];
            string b = queries[i][1];
            if (variables.find(a) == variables.end() || variables.find(b) == variables.end()) {
                ans[i] = -1.0;
                continue;
            }
            ans[i] = getAnswer(variables[a], variables[b], father, weight);
        }
        return ans;
    }

    int find(vector<int> &variables, vector<double> &weight, int x) {
        if (variables[x] != x) {
            int parent = find(variables, weight, variables[x]);
            weight[x] = weight[variables[x]] * weight[x]; 
            variables[x] = parent;
        }
        return variables[x];
    }

    // x 向 y
    void Union(vector<int> &variables, vector<double> &weight, int x, int y, double value) {
        int rootx = find(variables, weight, x);
        int rooty = find(variables, weight, y);
        if (rootx != rooty) {
            weight[rootx] = weight[variables[y]] * weight[rootx] * value;
            variables[rootx] = rooty;
        }
    }

    double getAnswer(int x, int y, vector<int> &variables, vector<double> &weight) {
        int rootx = find(variables, weight, x);
        int rooty = find(variables, weight, y);
        if (rootx != rooty) {
            return -1.0;
        } else {
            return weight[y] / weight[x];
        }
    }

    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (isConnected[i][j]) {
                    Union_Circle(father, i, j);
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (father[i] == i) {
                ans++;
            }
        }
        return ans;
    }

    int find_Circle(vector<int> &father, int x) {
        int parent = father[x];
        if (parent != x) {
            father[x] = find_Circle(father, parent);
        }
        return parent;
    }

    // x 向 y
    void Union_Circle(vector<int> &father, int x, int y) {
        int rootx = find_Circle(father, x);
        int rooty = find_Circle(father, y);
        if (rootx != rooty) {
            father[rootx] = rooty;
        }
    }

    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k % n;
        int cnt = __gcd(n, k);
        for (int i = 0; i < cnt; i++) {
            int start = i;
            int cur = nums[start];
            do {
                int temp = nums[(start + k) % n];
                nums[(start + k) % n] = cur;
                start = (start + k) % n;
                cur = temp;
            } while (start != i);
        }
    }

    int maxProfit(vector<int>& prices) {
        int buy1, buy2, sell1, sell2;
        int n = prices.size();
        buy1 = -prices[0];
        sell1 = 0;
        sell2 = 0;
        buy2 = 0x80000000;
        for (int i = 0; i < n; i++) {
            buy1 = max(buy1, -prices[i]);
            sell1 = max(sell1, buy1 + prices[i]);
            buy2 = max(buy2,sell1 -prices[i]);
            sell2 = max(sell2, buy2 + prices[i]);
        }
        return sell2;
    }

    vector<bool> prefixesDivBy5(vector<int>& A) {
        vector<bool> ans(A.size());
        int n = A.size();
        int num = A[0];
        ans[0] = A[0] == 0 ? true : false;
        if (A.size() == 1) {
            return ans;
        }
        for (int i = 1; i < n; i++) {
            num = ((num * 2) % 5 + A[i]) % 5;
            ans[i] = num == 0 ? true : false;
        }
        return ans;
    }

    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        for (int i = 0; i < n; i++) {
            if(!union_graph(father, edges[i][0], edges[i][1])) {
                return edges[i];
            }
        }
    }

    int find_graph(vector<int> &father, int x) {
        if (x != father[x]) {
            father[x] = find_graph(father, father[x]);
        }
        return father[x];
    }

    bool union_graph(vector<int> &father, int x, int y) {
        int rootx = find_graph(father, x);
        int rooty = find_graph(father, y);
        if (rootx == rooty) {
            return false;
        } else {
            father[rooty] = rootx;
            return true;
        }
    }

    int removeStones(vector<vector<int>>& stones) {
        int n = stones.size();
        int row[10001], col[10001];
        memset(row, -1, sizeof(row));
        memset(col, -1, sizeof(col));
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        for (int i = 0; i < n; i++) {
            union_stones(father, row, col, i, stones[i][0], stones[i][1]);
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (father[i] == i) {
                ans++;
            }
        }
        return n - ans;
    }

    int find_stones(vector<int> &father, int x) {
        if (x == -1) {
            return -1;
        }
        if (father[x] != x) {
            father[x] = find_stones(father, father[x]);
        }
        return father[x];
    }

    void union_stones(vector<int> &father, int row[], int col[], int index, int x, int y) {
        int rootx = find_stones(father, row[x]);
        int rooty = find_stones(father, col[y]);
        if (rootx == -1 && rooty == -1) {
            row[x] = index;
            col[y] = index;
            return;
        } else if (rootx == -1) {
            father[index] = rooty;
            row[x] = rooty;
        } else if (rooty == -1) {
            father[index] = rootx;
            col[y] = rootx;
        } else {
            father[index] = rootx;
            father[rooty] = rootx;
            col[y] = rootx;
        }
    }

    int countGoodRectangles(vector<vector<int>>& rectangles) {
        int ans = 0, max = 0;
        int n = rectangles.size();
        for (int i = 0; i < n; i++) {
            int side = min(rectangles[i][0], rectangles[i][1]);
            if (side > max) {
                max = side;
                ans = 1;
            } else if (side == max) {
                ans++;
            }
        }
        return ans;
    }

    int tupleSameProduct(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> map;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int res = nums[i] * nums[j];
                if (map.find(res) == map.end()) {
                    map[res] = 1;
                } else {
                    ans += map[res];
                    map[res]++;
                }
            }
        }
        ans *= 8;
        return ans;
    }

    vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
        // 考虑一下没有邮箱的情况
        int n = accounts.size();
        vector<int> father(n);
        unordered_map<string, int> map;
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        for (int i = 0; i < n; i++) {
            int anums = accounts[i].size();
            for (int j = 1; j < anums; j++) {
                // 没有出现过得邮箱
                if (map.find(accounts[i][j]) == map.end()) {
                    map[accounts[i][j]] = i;
                // 已经出现过的邮箱，需要合并
                } else {
                    union_account(father, i, map[accounts[i][j]]);
                }
            }
        }
        unordered_map<int, vector<string>> findq;
        int mapsize = map.size();
        for (auto &it : map) {
            int f = find_account(father, it.second);
            if (findq.find(f) == findq.end()) {
                vector<string> *pq = new vector<string>;
                pq->push_back(it.first);
                findq[f] = *pq;
            } else {
                findq[f].push_back(it.first);
            }
        }        
        int size = findq.size();
        vector<vector<string>> ans(size);
        int index = 0;
        for (auto &i : findq) {
            ans[index].push_back(accounts[i.first][0]);
            sort(i.second.begin(), i.second.end());
            for (auto &email : i.second) {
                ans[index].push_back(email);
            }
            index++;
        }
        return ans;
    }

    int find_account(vector<int> &father, int x) {
        if (father[x] != x) {
            father[x] = find_account(father, father[x]);
        }
        return father[x];
    }

    void union_account(vector<int> &father, int x, int y) {
        int rootx = find_account(father, x);
        int rooty = find_account(father, y);
        if (rootx != rooty) {
            father[rootx] = rooty;
        }
    }

    int minCostConnectPoints(vector<vector<int>>& points) {
        int n = points.size();
        auto dst = [&](int x, int y) -> int {
            return abs(points[x][0] - points[y][0]) + abs(points[x][1] - points[y][1]);
        };
        vector<Edge> edges;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                edges.emplace_back(dst(i, j), i, j);
            }
        }
        sort(edges.begin(), edges.end(), [](Edge a, Edge b) -> int {return a.len < b.len;});
        int cnt = 0, ans = 0, idx = 0;
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        while(cnt < n - 1) {
            if (union_point(father, edges[idx].x, edges[idx].y)) {
                cnt++;
                ans += edges[idx].len;
            }
            idx++;
        }
        return ans;
    }

    int find_point(vector<int> &father, int x) {
        if (father[x] != x) {
            father[x] = find_point(father, father[x]);
        }
        return father[x];
    }

    bool union_point(vector<int> &father, int x, int y) {
        int rx = find_point(father, x);
        int ry = find_point(father, y);
        if (rx == ry) {
            return false;
        } else {
            father[rx] = ry;
            return true;
        }
    }

    int maximumProduct(vector<int>& nums) {
        priority_queue<int, vector<int>, greater<int>> pq;
        priority_queue<int, vector<int>, less<int>> lq;
        int n = nums.size();
        pq.push(nums[0]);
        pq.push(nums[1]);
        pq.push(nums[2]);
        lq.push(nums[0]);
        lq.push(nums[1]);
        if (nums[2] < lq.top()) {
            lq.pop();
            lq.push(nums[2]);
        }
        for(int i = 3; i < n; i++) {
            if (nums[i] > pq.top()) {
                pq.pop();
                pq.push(nums[i]);
            }
            if (nums[i] < lq.top()) {
                lq.pop();
                lq.push(nums[i]);
            }
        }
        int nag[2];
        nag[0] = lq.top();
        lq.pop();
        nag[1] = lq.top();
        lq.pop();
        int ans = 1;
        if (nag[0] >= 0) {
            while(pq.size() != 0) {
                ans *= pq.top();
                pq.pop();
            }
        } else {
            int temp = nag[0] * nag[1];
            int post[3];
            for (int i = 0; i < 3; i++) {
                post[i] = pq.top();
                pq.pop();
                ans *= post[i];
            }
            if (ans < post[2] * temp) {
                ans = post[2] * temp;
            }
        }
        
        return ans;
    }

    vector<vector<int>> findCriticalAndPseudoCriticalEdges(int n, vector<vector<int>>& edges) {
        int m = edges.size();
        for (int i = 0; i < m; i++) {
            // 属于第几条边
            edges[i].push_back(i);
        }
        // Kruskal 给边排序
        sort(edges.begin(), edges.end(), [](auto &u, auto &v) { return u[2] < v[2];});
        int v = 0;
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        // 找到最小生成树的value 
        for (int i = 0; i < m; i++) {
            if (union_graph(father, edges[i][0], edges[i][1])) {
                v += edges[i][2];
            }
        }
        
        vector<vector<int>> ans(2);
        for (int i = 0; i < m; i++) {
            int value = 0;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                father[i] = i;
            }
            // 判断是否是关键边
            for (int j = 0; j < m; j++) {
                if (i != j && union_graph(father, edges[j][0], edges[j][1])) {
                    cnt++;
                    value += edges[j][2]; 
                }
                if (cnt != n - 1 || (cnt == n - 1 && value > v)) {
                    ans[0].push_back(edges[i][3]);
                    continue;
                }
            }
            for (int i = 0; i < n; i++) {
                father[i] = i;
            }
            value = 0;
            cnt = 0;
            // 伪关键边
            union_graph(father, edges[i][0], edges[i][1]);
            value += edges[i][2];
            cnt++;
            for (int j = 0; j < m; j++) {
                if (i != j && union_graph(father, edges[j][0], edges[j][1])) {
                    cnt++;
                    value += edges[j][2]; 
                }
                if (cnt == n - 1 && value == v) {
                    ans[1].push_back(edges[i][3]);
                }
            }
        }
        return ans;
    }

    vector<int> addToArrayForm(vector<int>& A, int K) {
        vector<int> num;
        while (K != 0) {
            num.push_back(K % 10);
            K /= 10;
        }
        int nA = A.size(), nK = num.size();
        vector<int> ans;
        int carry = 0;
        int temp = 0;
        reverse(A.begin(), A.end());
        int i = 0, j = 0;
        while (i < nA && j < nK) {
            temp = A[i++] + num[j++] + carry;
            carry = temp / 10;
            temp %= 10;
            ans.push_back(temp);
        }
        while (i < nA) {
            temp = A[i++] + carry;
            carry = temp / 10;
            temp %= 10;
            ans.push_back(temp);
        }
        while (j < nK) {
            temp = num[j++] + carry;
            carry = temp / 10;
            temp %= 10;
            ans.push_back(temp);
        }
        if (carry != 0) {
            ans.push_back(carry);
        }
        ans.reserve(ans.size())
        return ans;
    }

    int makeConnected(int n, vector<vector<int>>& connections) {
        int m = connections.size();
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        int useless = 0;
        int uni = n;
        for (int i = 0; i < m; i++) {
            if (!union_graph(father, connections[i][0], connections[i][1])) {
                useless++;
            } else {
                uni--;
            }
        }
        if (uni == 1) {
            return 0;
        }
        if (useless >= uni - 1) {
            return uni - 1;
        } else {
            return -1;
        }
    }

    int largestAltitude(vector<int>& gain) {
        int max = 0;
        int cur = 0, n = gain.size();
        for (int i = 0; i < n; i++) {
            cur += gain[i];
            if (cur >= max) {
                max = cur;
            }
        }
        return max;
    }

    int findLengthOfLCIS(vector<int>& nums) {
        int n = nums.size();
        int cur = 0x70000000;
        int start = -1;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (cur <= nums[i]) {
                cur = nums[i];
            } else {
                ans = max(ans, i - start);
                start = i;
            }
        }
        if (start != n - 1) {
            ans = max(ans, n - start);
        }
        return ans;
    }

    int minimumTeachings(int n, vector<vector<int>>& languages, vector<vector<int>>& friendships) {
        unordered_map<int, unordered_set<int> *> map;
        int m = languages.size();
        for (int i = 0; i < m; i++) {
            unordered_set<int> *set = new unordered_set<int>;
            for (int j = 0; j < languages[i].size(); j++) {
                set->insert(languages[i][j]);
            }
            map[i + 1] = set;
        }
        vector<vector<int>> need;
        int size = friendships.size();
        for (int i = 0; i < size; i++) {
            unordered_set<int> *u = map[friendships[i][0]];
            unordered_set<int> *v = map[friendships[i][1]];
            bool flag = false;
            for (int n : *u) {
                if ((*v).find(n) != (*v).end()) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                need.push_back(friendships[i]);
            }
        }
        int cnt = 0;
        int ans = 0x7fffffff;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < need.size(); j++) {
                if (map[need[j][0]]->find(i) == map[need[j][0]]->end()) {
                    cnt++;
                    map[need[j][0]]->insert(i);
                }
                if (map[need[j][1]]->find(i) == map[need[j][1]]->end()) {
                    cnt++;
                    map[need[j][1]]->insert(i);
                }
            }
            ans = min(ans, cnt);
        }
        return ans;
    }

    int numEquivDominoPairs(vector<vector<int>>& dominoes) {
        unordered_map<int, int> map;
        int ans;
        int n = dominoes.size();
        for (int i = 0; i < n; i++) {
            int a = dominoes[i][0] * 10 + dominoes[i][1];
            int b = dominoes[i][0] + dominoes[i][1] * 10;
            if (map.find(a) == map.end()) {
                map[a] = 1;
                map[b] = 1;
            } else {
                ans += map[a];
                map[a]++;
                if (a != b) {
                    map[b]++;
                }
                
            }
        }
        return ans;
    }

    int maxNumEdgesToRemove(int n, vector<vector<int>>& edges) {
        int size = edges.size();
        int ans = 0;
        int cnt = n;
        // 注意添加节点时需要-1
        vector<int> father(n);
        for (int i = 0; i < size; i++) {
            // type3
            if (edges[i][0] == 3) {
                if (!union_graph(father, edges[i][1]-1, edges[i][2]-1)) {
                    ans++;
                } else {
                    cnt--;
                }
            }
        }
        vector<int> ali(n);
        int cnt_ali = cnt;
        for (int i = 0; i < n; i++) {
            ali[i] = father[i];
        }
        for (int i = 0; i < size; i++) {
            // type1
            if (edges[i][0] == 1) {
                if (!union_graph(ali, edges[i][1]-1, edges[i][2]-1)) {
                    ans++;
                } else {
                    cnt_ali--;
                }
            }
        }
        if (cnt_ali != 1) {
            return -1;
        }
        vector<int> bob(n);
        int cnt_bob = cnt;
        for (int i = 0; i < n; i++) {
            bob[i] = father[i];
        }
        for (int i = 0; i < size; i++) {
            // type2
            if (edges[i][0] == 2) {
                if (!union_graph(bob, edges[i][1]-1, edges[i][2]-1)) {
                    ans++;
                } else {
                    cnt_bob--;
                }
            }
        }
        if (cnt_bob != 1) {
            return -1;
        } else {
            return ans;
        }
    }

    int pivotIndex(vector<int>& nums) {
        int n = nums.size();
        vector<int> sum(n+1);
        for (int i = 0; i < n; i++) {
            sum[i + 1] = sum[i] + nums[i];
        }
        for (int i = 0; i < n; i++) {
            int left = sum[i];
            int right = sum[n] - sum[i + 1];
            if (left == right) {
                return i;
            }
        }
        return -1;
    }

    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size();
        int left = 0, right = n * n - 1;
        bool map[n * n];
        memset(map, 0, sizeof(map));
        // cout << map[9] <<endl;
        flag = false;
        while (left < right) {
            int mid = (left + right) / 2;
            if (grid[0][0] > mid) {
                left = mid + 1;
                continue;
            }
            dfs(grid, map, 0, 0, mid);
            if (flag == true) {
                right = mid;
            } else {
                left = mid + 1;
            }
            // cout << flag <<endl;
            flag = false;
            memset(map, 0, sizeof(map));
        }
        return left;
    }

    void dfs(vector<vector<int>>& grid, bool map[], int i, int j, int target) {
        int n = grid.size();
        map[i * n + j] = true;
        if (i == n - 1 && j == n - 1) {
            flag = true;
            return;
        }
        if (i - 1 >= 0 && !map[(i - 1) * n + j] && grid[i - 1][j] <= target && !flag) {
            dfs(grid, map, i - 1, j, target);
        }
        if ((i + 1 < n) && (!map[(i + 1) * n + j]) && (grid[i + 1][j] <= target) && !flag) {
            dfs(grid, map, i + 1, j, target);
        }
        if (j - 1 >= 0 && !map[i * n + j - 1] && grid[i][j - 1] <= target && !flag) {
            dfs(grid, map, i, j - 1, target);
        }
        if (j + 1 < n && !map[i * n + j + 1] && grid[i][j + 1] <= target && !flag) {
            dfs(grid, map, i, j + 1, target);
        }
    }

    int numSimilarGroups(vector<string>& strs) {
        int n = strs.size();
        vector<vector<bool>> map(n);
        vector<int> father(n);
        for (int i = 0; i < n; i++) {
            father[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                similar(strs, map, i, j);
            }
        }
        int cnt = n;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (map[i][j]) {
                    if (union_graph(father, i, j)) {
                        cnt--;
                    }
                }
            }
        }
        return cnt;
    }

    void similar(vector<string> &strs, vector<vector<bool>> &map, int i, int j) {
        int len = strs[i].size();
        if (len != strs[j].size()) {
            map[i][j] = false;
            return;
        }
        int cnt = 0;
        for (int k = 0; k < len; k++) {
            if (strs[i][k] != strs[j][k]) {
                cnt++;
                if (cnt > 2) {
                    map[i][j] = false;
                    return;
                }
            }
        }
        map[i][j] = true;
    }
    
};


int main() {
    // vector<int> cost = vector<int>();
    // cout<< Solution().minCostClimbingStairs(cost) <<endl;
    // vector<int> nums = {1,3,-1,-3,5,3,6,7};
    // int k = 3;
    // vector<int> ans = Solution().maxSlidingWindow_monotonic(nums, k);
    // for(int i=0; i<ans.size(); i++) {
    //     cout<< ans[i] << " ";
    // }
    // string s = "abbxxxxzzy";
    // vector<vector<int>> ans = Solution().largeGroupPositions(s);
    // for(auto &x : ans) {
    //     for(auto &i : x) {
    //         cout<< i << " ";
    //     }
    //     cout << endl;
    // }
    return 0;
}