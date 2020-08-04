import jdk.jshell.execution.Util;
import org.w3c.dom.ls.LSException;

import javax.print.DocFlavor;
import java.awt.event.MouseAdapter;
import java.lang.reflect.Array;
import java.util.*;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) {val = x;}
}

class Pair<K, V> {
    K k;
    V v;
    Pair(K kk, V vv) {k = kk; v = vv;}
}


class Solution {
    private static final int uncolored = 0;
    private static final int red = 1;
    private static final int green = 2;

    public void test() {
//        int[] A = {0,1,1,1,1};
//        int[] B = {1,0,1,0,1};
//        System.out.println(findLength(A, B));
//        int[] nums = {3,2};
//        int target = 2;
//        System.out.println(search(nums, target));
//        String haystack = "aaa";
//        String needle = "aaaa";
//        System.out.println(strStr(haystack, needle));
//        int[][] matrix = {{1,  5,  9}, {10, 11, 13}, {12, 13, 15}};
//        int k=8;
//        System.out.println(kthSmallest(matrix, k));
//        int[] nums = {-10,-3,0,5,9};
//        TreeNode root = sortedArrayToBST(nums);
//        System.out.println(root);
//        int[][] matrix = {{1,3,5,7},{10,11,16,20},{23,30,34,50}};
//        int target = 10;
//        System.out.println(binarysearch(matrix, 0, target));
//        System.out.println(searchMatrix1(matrix, target));
//        String s1 = "(()";
//        String s2 = "()())";
//        String s3 = ")()(())";
//        System.out.println(longestValidParentheses(s1));
//        System.out.println(longestValidParentheses(s2));
//        System.out.println(longestValidParentheses(s3));
//        int[][] mat = {{1,0,1},{1,1,0},{1,1,0}};
//        System.out.println(numSubmat(mat));
//        TreeNode root = new TreeNode(1);
//        root.left = new TreeNode(2);
//        System.out.println(hasPathSum(root, 1));
//        int[] target = {2,3,4,5,8,9,10};
//        int n=10;
//        List<String> res = buildArray(target, n);
//        for(String i : res) {
//            System.out.println(i);
//        }
//        int shorter = 1;
//        int longer = 2;
//        int k = 3;
//        int[] res = divingBoard(shorter, longer, k);
//        for(int i : res) {
//            System.out.print(i+" ");
//        }
//        String S = "bbbextm";
//        String T = "bbb#extm";
//        System.out.println(backspaceCompare(S, T));
//        String[] dictinary = {"just"};
//        String sentence = "sjuste";
//        System.out.println(respace(dictinary, sentence));
//        String date = "6th Jun 1933";
//        System.out.println(reformatDate(date));
//        int[] nums = {1,2,3,4};
//        System.out.println(rangeSum(nums, 4, 1, 10));
//        int[] nums = {1,5,6,14,15};
//        System.out.println(minDifference(nums));
//        int n = 8;
//        System.out.println(winnerSquareGame(n));
//        int[] nums1 = {4,9,5};
//        int[] nums2 = {9,4,9,8,4};
//        int[] res = intersect(nums1, nums2);
//        for (int i : res) {
//            System.out.println(i);
//        }
//        List<Integer> l1 = new LinkedList<>();
//        l1.add(2);
//        List<Integer> l2 = new LinkedList<>();
//        l2.add(3);l2.add(4);
//        List<Integer> l3 = new LinkedList<>();
//        l3.add(6);l3.add(5);l3.add(7);
//        List<Integer> l4 = new LinkedList<>();
//        l4.add(4);l4.add(1);l4.add(8);l4.add(3);
//        List<List<Integer>> triangle = new LinkedList<>();
//        triangle.add(l1);
//        triangle.add(l2);
//        triangle.add(l3);
//        triangle.add(l4);
//        System.out.println(minimumTotal(triangle));
//        int[] nums = {1,3,5,6};
//        int target = 0;
//        System.out.println(searchInsert(nums, target));
//        int[][] graph = {{1,2,3}, {0,2}, {1,3}, {0,2}};
//        System.out.println(isBipartite(graph));
//        int numBottles=15, numExchange = 4;
//        System.out.println(numWaterBottles(numBottles, numExchange));
//        int[] nums = {3,1,5,8};
//        System.out.println(maxCoins(nums));
//        int[] nums = {2,5,6,0,0,1,2};
//        int target = 0;
//        System.out.println(search2(nums, target));
//        int n = 7;
//        int[][] edges = {{0,1}, {0,2}, {1,4}, {1,5}, {2,3}};
//        String labels = "abaedcd";
//        int[] val = countSubTrees(n, edges, labels);
//        for(int value : val) {
//            System.out.print(value + " ");
//        }
        String S = "aaaa";
        String T = "bbaaa";
        System.out.println(isSubsequence(S, T));
    }

    public int findLength(int[] A, int[] B) {
        if(A.length == 0 || B.length == 0) {
            return 0;
        }
        int res = 0;
        int[][] dp = new int[A.length][B.length];
        for(int i=0; i<A.length; i++) {
            if(A[i] == B[0]) {
                dp[i][0] = 1;
                res = 1;
            }
            //System.out.print(dp[i][0]+" ");
        }
        //System.out.println();
        for(int j=0; j<B.length; j++) {
            if(B[j] == A[0]) {
                dp[0][j] = 1;
                res = 1;
            }
            //System.out.print(dp[0][j]+" ");
        }
        //System.out.println();
        for(int i=1; i<A.length; i++) {
            for(int j=1; j<B.length; j++) {
                if(A[i] == B[j]) {
                    dp[i][j] = Math.max(dp[i][j], dp[i-1][j-1]+1);
                }
                res = Math.max(res, dp[i][j]);
                //System.out.print(dp[i][j]+" ");
            }
            //System.out.println();
        }
//        for(int i=0; i<A.length; i++) {
//            for(int j=0; j<B.length; j++) {
//                System.out.print(dp[i][j] + " ");
//            }
//            System.out.println();
//        }
        return res;
    }

    public int search(int[] nums, int target) {
        if(nums.length == 0) {
            return -1;
        }
        int left = 0, right = nums.length-1;
        int mid = (left+right) / 2;
        //顺序的情况
        if(nums[left] < nums[right]) {
            while(left <= right) {
                mid = (left+right) / 2;
                if(nums[mid] == target)
                    return mid;
                else if(target > nums[mid]) {
                    left = mid+1;
                }
                else right = mid-1;
            }
        }
        else {
            while (left <= right) {
                mid = (left+right)/2;
                if(nums[mid] == target) {
                    return mid;
                }
                //需要继续搜索的情况
                //在mid的左边是有序的
                if(nums[0] <= nums[mid]) {
                    //判断是否在有序序列中
                    if (target > nums[mid] || target < nums[0]) {
                        left = mid + 1;
                    }
                    else {
                        right = mid - 1;
                    }
                }
                //mid的右侧是有序的
                //此处有一点需要注意，旋转后总是有一段小的在右侧
                else {
                    if(target > nums[mid] && target <= nums[right]) {
                        left = mid + 1;
                    }
                    else {
                        right = mid - 1;
                    }
                }
            }
        }
        if(nums[mid] == target)
            return mid;
        return -1;
    }

    public int strStr(String haystack, String needle) {
        if(needle.length() == 0) {
            return 0;
        }
        int h = haystack.length();
        int n = needle.length();
        for(int i=0; i<haystack.length(); i++) {
            if(haystack.charAt(i) == needle.charAt(0)) {
                int j=1;
                while(i+j < h && j< n) {
                    if(haystack.charAt(i+j) != needle.charAt(j))
                        break;
                    j++;
                }
                if(j == n)
                    return i;
            }
        }
        return -1;
    }

    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        int left = matrix[0][0];
        int right = matrix[n-1][n-1];
        int mid = (left+right)/2;
        while(left < right) {
            mid = (left+right)/2;
            if(check(matrix, mid, k, n)) {
                right = mid;
            }
            else {
                left = mid+1;
            }
        }
        return left;
    }

    public boolean check(int[][] matrix, int mid, int k, int n) {
        int i = n-1;
        int j = 0;
        int sum = 0;
        while(i>=0 && j < n) {
            if(matrix[i][j] <= mid) {
                j++;
                sum += i+1;
            }
            else {
                i--;
            }
        }
        return sum >= k;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return initTree(nums, 0, nums.length);
    }

    public TreeNode initTree(int[] nums, int left, int right) {
        if(left == right) {
            return null;
        }
        else {
            int mid = (left+right)/2;
            TreeNode cur = new TreeNode(nums[mid]);
            cur.left = initTree(nums, left, mid );
            cur.right = initTree(nums, mid+1, right);
            return cur;
        }
    }

    public boolean searchMatrix1(int[][] matrix, int target) {
        int cnt=0;
        int n = matrix.length;
        if(n == 0) {
            return false;
        }
        int m = matrix[0].length;
        if(m == 0) {
            return false;
        }
        while(cnt < n && matrix[cnt][m-1] < target) {
            cnt++;
        }
        if(cnt == n) {
            return false;
        }
        if(matrix[cnt][m-1] == target)
            return true;
        if(cnt == 0) {
            return binarysearch(matrix, 0, target);
        }
        else {
            return binarysearch(matrix, cnt-1, target) || binarysearch(matrix, cnt, target);
        }
    }

    public boolean binarysearch(int[][] matrix, int row, int target) {
        int left=0, right=matrix[0].length;
        while(left<right) {
            int mid = (left+right)/2;
            if(matrix[row][mid] == target)
                return true;
            if(target > matrix[row][mid]) {
                left = mid+1;
            }
            else {
                right = mid;
            }
        }
        return false;
    }

    public int longestValidParentheses(String s) {
        Stack<Pair<Integer, Character>> stack = new Stack<>();
        int res = 0;
        for(int i=0; i<s.length(); i++) {
            if(stack.empty()) {
                stack.push(new Pair<>(i, s.charAt(i)));
                continue;
            }
            Pair<Integer, Character> top = stack.peek();
            if(top.v == '(' && s.charAt(i) == ')') {
                stack.pop();
                if(stack.empty()) {
                    res = Math.max(res, i+1);
                }
                else {
                    res = Math.max(res, i - stack.peek().k);
                }
            }
            else {
                stack.push(new Pair<>(i, s.charAt(i)));
            }
        }
        return res;
    }

    public boolean canMakeArithmeticProgression(int[] arr) {
        if(arr.length <= 2) {
            return true;
        }
        Arrays.sort(arr);
        int diff = 0;
        for(int i=0; i<arr.length-1; i++) {
            if(i == 0) {
                diff = arr[1]-arr[0];
            }
            if(diff != arr[i+1] - arr[i]) {
                return false;
            }
        }
        return true;
    }

    public int getLastMoment(int n, int[] left, int[] right) {
        int max_left = 0;
        for(int i=0; i<left.length; i++) {
            if(left[i] > max_left) {
                max_left = left[i];
            }
        }
        int min_right = Integer.MAX_VALUE;
        for(int i=0; i<right.length; i++) {
            if(right[i] < min_right) {
                min_right = right[i];
            }
        }
        return Math.max(max_left, n-min_right);
    }

    public int numSubmat(int[][] mat) {
        int[][] sum = new int[mat.length+1][mat[0].length+1];
        for(int i=1; i<=mat.length; i++) {
            for(int j=1; j<=mat[0].length; j++) {
                sum[i][j] = sum[i][j-1] + mat[i-1][j-1];
            }
        }
        int res = 0;
        for(int k=1; k<=mat[0].length; k++) {
            for(int i=1; i<=mat.length; i++) {
                for(int j=k; j<=mat[0].length; j++) {
                    if(sum[i][j]-sum[i][j-k] == k) {
                        res++;
                    }
                }
            }
        }
        for(int k=2; k<=mat.length; k++) {
            for(int i=1; i<=mat[0].length; i++) {
                for(int j=k; j<=mat.length; j++) {
                    if(sum[j][i]-sum[j-k][i] == k) {
                        res++;
                    }
                }
            }
        }
        return res;
    }

    public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[s.length()+1][p.length()+1];
        dp[0][0] = true;
        for(int i=1; i<=p.length(); i++) {
            if(p.charAt(i-1) == '*')
                dp[0][i] = true;
            else break;
        }
        for(int i=1; i<=s.length(); i++) {
            for(int j=1; j<=p.length(); j++) {
                if(p.charAt(j-1) == '*') {
                    dp[i][j] = dp[i-1][j] || dp[i][j-1];
                }
                if(s.charAt(i-1) == p.charAt(j-1) || p.charAt(j-1)=='?') {
                    dp[i][j] = dp[i-1][j-1];
                }
            }
        }
        return dp[s.length()][p.length()];
    }

    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) {
            return false;
        }
        return findleaf(root, 0, sum);
    }

    public boolean findleaf(TreeNode root, int sum, int target) {
        if(root == null) {
            return false;
        }
        if(root.left == null && root.right == null) {
            if(sum+root.val== target)
                return true;
            return false;
        }
        return findleaf(root.left, sum+root.val, target) ||
                findleaf(root.right, sum+root.val, target);
    }

    public List<String> buildArray(int[] target, int n) {
        List<String> res = new LinkedList<>();
        int num = 0;
        int i=0, j=1;
        while(j <= n && i < target.length) {
            res.add("Push");
            if (j != target[i]) {
                res.add("Pop");
                j++;
            } else {
                j++;
                i++;
            }
        }
        return  res;
    }

    public int[] divingBoard(int shorter, int longer, int k) {
        if(k == 0) {
            return new int[0];
        }
        if(shorter == longer) {
            int[] res = new int[1];
            res[0] = shorter*k;
            return res;
        }
        int[] res = new int[k+1];
        for (int i = 0; i <= k; i++) {
            res[i] = (k - i) * shorter + i * longer;
        }
        return res;
    }

    public boolean backspaceCompare(String S, String T) {
        int i=S.length()-1, j=T.length()-1;
        int skipS=0, skipT=0;
        while(i>=0 || j>=0) {
            while(i >= 0) {
                if(S.charAt(i) == '#') {
                    skipS++;
                    i--;
                }
                else if(skipS > 0) {
                    skipS--;
                    i--;
                }
                else break;
            }
            while(j >= 0) {
                if(T.charAt(j) == '#') {
                    skipT++;
                    j--;
                }
                else if(skipT > 0) {
                    skipT--;
                    j--;
                }
                else break;
            }
            if(i>=0 && j>=0 && S.charAt(i) != T.charAt(j)) {
                return false;
            }
            if((i >= 0) != (j >= 0)) {
                return false;
            }
            i--; j--;
        }
        return true;
    }

    public int respace(String[] dictionary, String sentence) {
        Set<String> dict = new HashSet<>(Arrays.asList(dictionary));
        int n = sentence.length();
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = dp[i - 1] + 1;
            for (int idx = 0; idx < i; idx++) {
                if (dict.contains(sentence.substring(idx, i))) {
                    dp[i] = Math.min(dp[i], dp[idx]);
                }
            }
        }
        return dp[n];
    }

    public String reformatDate(String date) {
        StringBuffer sb = new StringBuffer();
        String[] Date = date.split(" ");
        String[] Month = {"Jan", "Feb", "Mar", "Apr", "May",
                "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        int year = Integer.parseInt(Date[2]);
        int month = 0;
        int day = Integer.parseInt(Date[0].substring(0, Date[0].length()-2));
        for(int i=0; i<Month.length; i++) {
            if(Month[i].compareTo(Date[1]) == 0) {
                month = i+1;
                break;
            }
        }
        sb.append(year);
        sb.append("-");
        if(month < 10) {
            sb.append("0");
        }
        sb.append(month);
        sb.append("-");
        if(day < 10) {
            sb.append("0");
        }
        sb.append(day);
        return sb.toString();
    }

    public int rangeSum(int[] nums, int n, int left, int right) {
        int[] sum = new int[n*(n+1)/2];
        int index = 0;
        for(int i=0; i<n; i++) {
            int temp = nums[i];
            sum[index++] = temp;
            for(int j=i+1; j<n; j++) {
                temp += nums[j];
                sum[index++] = temp;
            }
        }
        Arrays.sort(sum);
        int ans = 0;
        for(int i=left-1; i<right; i++) {
            ans = (ans + sum[i]) % 1000000007;
        }
        return ans;
    }

    public int minDifference(int[] nums) {
        if(nums.length <= 4) {
            return 0;
        }
        Arrays.sort(nums);
        int ans = Integer.MAX_VALUE;
        int n = nums.length-1;
        for(int i=0; i<4; i++) {
            ans = Math.min(nums[n-3+i] - nums[i], ans);
        }
        return ans;
    }

    public boolean winnerSquareGame(int n) {
        int i=1;
        while(i*i <= n) {
            if(dfs(n - i*i, 0)) {
                return true;
            }
            i++;
        }
        return false;
    }

    public boolean dfs(int remain, int depth) {
        if(remain == 0) {
            return depth % 2 == 0;
        }
        else {
            int i=1;
            while(i*i <= remain) {
                if(!dfs(remain-i*i, depth+1)) {
                    return false;
                }
                i++;
            }
            return true;
        }
    }

    public int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for(int i : nums1) {
            int cnt = map.getOrDefault(i, 0) + 1;
            map.put(i, cnt);
        }
        List<Integer> list = new LinkedList<>();
        for(int i : nums2) {
            int value = map.getOrDefault(i, 0);
            if(value > 0) {
                map.put(i, value-1);
                list.add(i);
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        int[] dp = new int[triangle.size()];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[dp.length-1] = 0;
        for(List<Integer> list : triangle) {
            int idx = dp.length - list.size();
            for(int num : list) {
                if(idx == dp.length-1) {
                    dp[idx] = dp[idx] + num;
                    continue;
                }
                dp[idx] = Math.min(dp[idx], dp[idx+1]) + num;
                idx++;
            }
        }
        int res = Integer.MAX_VALUE;
        for(int i : dp) {
            if(res > i) {
                res = i;
            }
        }
        return res;
    }

    public int searchInsert(int[] nums, int target) {
        if(nums.length == 0) {
            return 0;
        }
        int left=0, right = nums.length;
        int mid = 0;
        while(left < right) {
            mid = (left+right)/2;
            if(target == nums[mid]) {
                return mid;
            }
            else if (nums[mid] < target) {
                left= mid+1;
            }
            else {
                right = mid;
            }
        }
        if(target > nums[mid]) {
            return mid+1;
        }
        else {
            return mid;
        }
    }

    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        boolean flag = true;
        int[] color = new int[n];
        for(int i=0; i<n&&flag; i++) {
            if(color[i] == uncolored) {
                flag = dfs(color, i, red, graph);
            }
        }
        return flag;
    }

    public boolean dfs(int[] color, int node, int c, int[][] graph) {
        color[node] = c;
        int tobedyed = c == red ? green : red;
        for(int i : graph[node]) {
            if(color[i] == uncolored) {
                 if(!dfs(color, i, tobedyed, graph))
                     return false;
            }
            else if(color[i] != tobedyed) {
                return false;
            }
        }
        return true;
    }

    public int numWaterBottles(int numBottles, int numExchange) {
        int res = numBottles;
        int emptybootles = res;
        while(emptybootles >= numExchange) {
            res += emptybootles/numExchange;
            emptybootles = emptybootles%numExchange + emptybootles/numExchange;
        }
        return res;
    }

    public int maxCoins(int[] nums) {
        int[] val = new int[nums.length + 2];
        val[0] = 1;
        val[nums.length+1] = 1;
        for(int i=1; i<=nums.length; i++) {
            val[i] = nums[i-1];
        }
        int[][] map = new int[nums.length+2][nums.length+2];
        for(int i=0; i<map.length; i++) {
            Arrays.fill(map[i], -1);
        }
        return solve(map, val, 0, val.length-1);
    }

    public int solve(int[][] map, int[] val, int left, int right ) {
        if(left >= right - 1) {
            return 0;
        }
        if(map[left][right] != -1) {
            return map[left][right];
        }
        for(int i=left+1; i<right; i++) {
            int sum = val[left] * val[i] * val[right];
            sum += solve(map, val, left, i) + solve(map, val, i, right);
            map[left][right] = Math.max(sum, map[left][right]);
        }
        return map[left][right];
    }

    public int[] twoSum(int[] numbers, int target) {
        int[] res = new int[2];
        for(int i=0; i<numbers.length; i++) {
            for(int j=i+1; j<numbers.length; j++) {
                if(target == numbers[i] + numbers[j]) {
                    res[0] = i+1;
                    res[1] = j+1;
                    break;
                }
            }
        }
        return res;
    }

    public boolean search2(int[] nums, int target) {
        int left = 0, right = nums.length-1;
        if(nums.length == 0) return false;
        if(nums.length == 1) return nums[0] == target;
        while(left <= right) {
            int mid = (left + right) / 2;
            if(target == nums[mid]) return true;
            //必须为严格递增才可以进行舍去
            //判断mid在哪一段有序序列中
            if(nums[left] < nums[mid]) {
                if(target < nums[mid] && target >= nums[left]) {
                    right = mid - 1;
                }
                else {
                    left = mid + 1;
                }
            }
            else if(nums[mid] < nums[right]) {
                if(target <= nums[right] && target > nums[mid]) {
                    left = mid + 1;
                }
                else {
                    right = mid - 1;
                }
            }
            else {
                if(nums[left] == nums[mid]) {
                    left++;
                }
                if(nums[right] == nums[mid]) {
                    right--;
                }
            }
        }
        return false;
    }

    public List<TreeNode> generateTrees(int n) {
        if(n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> allTrees = new LinkedList<>();
        if(start > end) {
            allTrees.add(null);
            return allTrees;
        }
        //枚举可行根节点
        for(int i=start; i<=end; i++) {
            //以i为根节点
            List<TreeNode> leftTrees = generateTrees(start, i-1);
            List<TreeNode> rightTrees = generateTrees(i+1, end);
            for(TreeNode left : leftTrees) {
                for(TreeNode right : rightTrees) {
                    TreeNode temp = new TreeNode(i);
                    temp.left = left;
                    temp.right = right;
                    allTrees.add(temp);
                }
            }
        }
        return allTrees;
    }

    public int[] countSubTrees(int n, int[][] edges, String labels) {
        boolean[][] graph = new boolean[n][n];
        for(int[] edge : edges) {
            graph[edge[0]][edge[1]] = true;
        }
        int[] val = new int[n];
        Arrays.fill(val, 0);
        for(int i=0; i<n && val[i] == 0; i++) {
            dfs(graph, val, labels, i);
        }
        return val;
    }

    public void dfs(boolean[][] graph, int[] val, String labels, int node) {
        int n = graph.length;
        val[node] = 1;
        boolean isleaf = false;
        for(int i=0; i<n; i++) {
            if(graph[node][i]) {
                //有连接，不算是叶子节点
                isleaf = true;
                //还未遍历到
                if(val[i] == 0) {
                    dfs(graph, val, labels, i);
                }
                //已经遍历过
                if(labels.charAt(node) == labels.charAt(i)) {
                    val[node] += val[i];
                }
            }
        }
    }

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m+1][n+1];
        for(int i=0; i<=m; i++) {
            Arrays.fill(dp[i], Integer.MAX_VALUE);
        }
        dp[0][1] = 0;
        for(int i=1; i<=m; i++) {
            for(int j=1; j<=n; j++) {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i-1][j-1];
            }
        }
        return dp[m][n];
    }

    public boolean divisorGame(int N) {
        boolean[] dp = new boolean[N+1];
        dp[1] = false;
        for(int i=1; i<N && N%i==0; i++) {
            int temp = i + i;
            while(temp <= N) {
                if(!dp[temp]) {
                    dp[temp] = !dp[temp - i];
                }
                temp += i;
            }
        }
        return dp[N];
    }

    public boolean isSubsequence(String s, String t) {
        int s_point = 0, t_point = 0;
        while(s_point < s.length() && t_point < t.length()) {
            while(t_point < t.length() && s.charAt(s_point) != t.charAt(t_point) ) {
                t_point++;
            }
            if(t_point == t.length())
                return false;
            s_point++;
            t_point++;
        }
        return s_point == s.length();
    }
    

}


public class Main {
    public static void main(String[] args) {
        Solution s = new Solution();
        s.test();
    }
}
