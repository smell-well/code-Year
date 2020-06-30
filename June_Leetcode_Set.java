import java.lang.annotation.ElementType;
import java.lang.annotation.Target;
import java.lang.reflect.Array;
import java.nio.charset.IllegalCharsetNameException;
import java.util.*;

class Pair {
    int K;
    int V;
    Pair(int k, int v) {K = k; V = v;}
}

class ListNode {
    int val;
    ListNode next;
    ListNode(int x) {val = x;}
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x;}
}

class SubrectangleQueries {
    public int[][] map = null;

    public SubrectangleQueries(int[][] rectangle) {
        map = new int[rectangle.length][rectangle[0].length];
        for(int i=0; i<rectangle.length; i++) {
            for(int j=0; j<rectangle[0].length; j++) {
                map[i][j] = rectangle[i][j];
            }
        }
    }

    public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
        for(int i=row1; i<=row2; i++) {
            for(int j=col1; j<=col2; j++) {
                map[i][j] = newValue;
            }
        }
    }

    public int getValue(int row, int col) {
        return map[row][col];
    }
}

class Codec {
    public String serialize(TreeNode root) {
        StringBuffer sb = new StringBuffer();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode temp = queue.remove();
            if(temp != null) {
                sb.append(String.valueOf(temp.val));
                sb.append(',');
                queue.add(temp.left);
                queue.add(temp.right);
            }
            else {
                sb.append("n,");
            }
        }
        sb.delete(sb.length()-1, sb.length());
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] nums = data.split(",");
        TreeNode root = null;
        Queue<TreeNode> queue = new LinkedList<>();
        if(nums[0].compareTo("n") == 0) {
            return null;
        }
        else {
            root = new TreeNode(Integer.parseInt(nums[0]));
            queue.add(root);
        }
        for(int i=1; i<nums.length; i+=2) {
            TreeNode parent = queue.remove();
            if(nums[i].compareTo("n") != 0) {
                parent.left = new TreeNode(Integer.parseInt(nums[i]));
                queue.add(parent.left);
            }
            if(nums[i+1].compareTo("n") != 0) {
                parent.right = new TreeNode(Integer.parseInt(nums[i+1]));
                queue.add(parent.right);
            }
        }
        return root;
    }
}

class CQueue {
    Stack<Integer> head = null;
    Stack<Integer> temp = null;

    public CQueue() {
        head = new Stack<>();
        temp = new Stack<>();

    }

    public void appendTail(int value) {
        head.add(value);
    }

    public int deleteHead() {
        if(head.isEmpty()) {
            return -1;
        }
        while(!head.isEmpty()) {
            temp.add(head.pop());
        }
        int res = temp.pop();
        while(!temp.isEmpty()) {
            head.add(temp.pop());
        }
        return res;
    }
}

class Solution {
    public void test() {
        //int[] nums = {2, 1, 1, 2};
        //System.out.println(rob(nums));
//        String s = "00110110";
//        int k = 2;
//        System.out.println(hasAllCodes(s, k));
//        int[][] prerequisites = {{4,3}, {4,1}, {4,0}, {3,2},{3,1},{3,0},{2,1},{2,0},{1,0}};
//        int[][] queries = {{1,4},{4,2},{0,1},{4,0},{0,2},{1,3},{0,1}};
//        List<Boolean> res = checkIfPrerequisite(5, prerequisites, queries);
//        for(boolean i : res) {
//            System.out.println(i?1:0);
//        }
//        int N = 6, K = 1, W = 10;
//        System.out.println(new21Game(N,K,W));
//        int[] nums = {1,2,3,4};
//        nums = productExceptSelf(nums);
//        for(int num : nums) {
//            System.out.println(num);
//        }
//        int[][] matrix = {{1,2,3,4}, {5,6,7,8}, {9,10,11,12,}};
//        //int[][] matrix = {{1,2,3}, {5,6,7}, {9,10,11}};
//        //int[][] matrix = {{1,2,3}};
//        //int[][] matrix = {{1},{2},{3}};
//        List<Integer> res = spiralOrder(matrix);
//        for(int i : res) {
//            System.out.print(i + " ");
//       }
//        String[] equations = {"a==b", "b!=c", "c==a"};
//        System.out.println(equationsPossible(equations));
//        int num = 12258;
//        System.out.println(translateNum(num));
//        ListNode head = new ListNode(1);
//        ListNode temp = head;
//        for(int i=1; i<5; i++) {
//            temp.next = new ListNode(i+1);
//            temp = temp.next;
//        }
//        temp.next = null;
//        head = reverseBetween(head, 1, 4);
//        temp = head;
//        while(temp != null) {
//            System.out.print(temp.val + " ");
//            temp = temp.next;
//        }
//        int[] T = {73, 74, 75, 71, 69, 72, 76, 73};
//        T = dailyTemperatures(T);
//        for (int i : T) {
//            System.out.print(i + " ");
//        }
//        int[] square1 = {249, -199, 5};
//        int[] square2 = {-1, 136, 76};
//        double[] res = cutSquares(square1, square2);
//        for(double i : res) {
//            System.out.print(i + " ");
//        }
//        int[] nums = {-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6};
//        List<List<Integer>> res = threeSum(nums);
//        for(List<Integer> i : res) {
//            for(int j : i) {
//                System.out.print(j + " ");
//            }
//            System.out.println();
//        }
//        int[] prices = {8,4,6,2,3};
//        prices = finalPrices(prices);
//        for(int price : prices) {
//            System.out.print(price + " ");
//        }
//        int[][] rect = {{1,2,1}, {4,3,4}, {3,2,1}, {1,1,1}};
//        SubrectangleQueries obj = new SubrectangleQueries(rect);
//        obj.updateSubrectangle(0,0,3,2,5);
//        System.out.println(obj.getValue(0,2));
//        int[] arr = {43,5,4,4,37,5,3,3,42,9,1,4,30,18,24,4,11,11,1,1,52,33,7,8,2,1,1,21,6};
//        int target = 52;
//        System.out.println(minSumOfLengths(arr, target));
//        int[] arr = {2,3,5};
//        int target = 10;
//        System.out.println(findBestValue(arr, target));
//        String[] strs = {"dog","racecar","car"};
//        System.out.println(longestCommonPrefix(strs));
//        TreeNode root = new TreeNode(-1);
//        root.left = new TreeNode(0);
//        root.right = new TreeNode(1);
//        root.left = new TreeNode(2);
//        root.right = new TreeNode(3);
//        root.left.left = new TreeNode(6);
//        root.left.right = null;
//        root.right.left = new TreeNode(4);
//        root.right.right = new TreeNode(5);
//        Codec codec = new Codec();
//        System.out.println(codec.serialize(codec.deserialize(codec.serialize(root))));
//        int[] A = {8,1,5,2,6};
//        System.out.println(maxScoreSightseeingPair(A));
//        String pattern = "bbba";
//        String value = "xxxxxxy";
//        System.out.println(patternMatching(pattern, value));
//        String a = "1010";
//        String b = "1011";
//        System.out.println(addBinary(a, b));
//        int[] nums =  {0,2,1,-3};
//        int target = 1;
//        System.out.println(threeSunClosest(nums, target));
//        String s = "applepenapple";
//        List<String> wordDict = new LinkedList<>();
//        wordDict.add("apple");
//        wordDict.add("pen");
//        System.out.println(wordBreak(s, wordDict));
//        String s = " ";
//        System.out.println(lengthOfLongestSubstring(s));
//        int[] nums = {0,1,2};
//        System.out.println(firstMissingPositive(nums));
//        int[] salary = {6000,5000,4000,3000,2000,1000};
//        System.out.println(average(salary));
//        int n=4, k=4;
//        System.out.println(kthFactor(n, k));
//        int[] nums = {0,1,1,1,0,1,1,0,1};
//        System.out.println(longestSubarray(nums));
//        int[] nums = {2,3,1,2,4,3};
//        int s = 7;
//        System.out.println(minSubArrayLen(s, nums));
//        int[] arr = {0,1};
//        int start = 1;
//        System.out.println(canReach(arr, start));
//        int[] nums = {3,2,3,1,2,4,5,5,6};
//        int k = 4;
//        System.out.println(findKthLargest(nums, k));
        int[] nums = {4,5,6,7,0,1,2};
        System.out.println(findMin(nums));
    }
    public int rob(int[] nums) {
        int[] dp = new int[nums.length];
        if(nums.length <= 2) {
            int max = 0;
            for(int i : nums) {
                if(i > max)
                    max = i;
            }
            return max;
        }
        else {
            dp[0] = nums[0];
            dp[1] = nums[1];
            for(int i=2; i<nums.length; i++) {
                int max = 0;
                for(int j=0; j<=i-2; j++) {
                    if(dp[j] > max) {
                        max = dp[j];
                    }
                }
                dp[i] = Math.max(max + nums[i], dp[i-1]);
            }
            int max = 0;
            for(int i : dp) {
                if(i > max)
                    max = i;
            }
            return max;
        }

    }

    public boolean canBeEqual(int[] target, int[] arr) {
        int[] count = new int[1000];
        for(int i : target) {
            count[i]++;
        }
        for(int i : arr) {
            count[i]--;
        }
        for(int i=0; i<1000; i++) {
            if(count[i]<0 || count[i]>0) {
                return false;
            }
        }
        return true;
    }

    public boolean hasAllCodes(String s, int k) {
        HashSet<String> set = new HashSet<>();
        for(int i=0; i<=s.length()-k; i++) {
            set.add(s.substring(i, i+k));
        }
        return !(set.size() < Math.pow(2, k));
    }

    public List<Boolean> checkIfPrerequisite(int n, int[][] prerequisites, int[][] queries) {
        boolean[][] map = new boolean[n][n];
        for(int i=0; i<prerequisites.length; i++) {
            map[prerequisites[i][0]][prerequisites[i][1]] = true;
        }
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                for(int k=0; k<n; k++) {
                    if(map[i][j] && map[j][k]) {
                        map[i][k] = true;
                    }
                }
            }
        }
        for(int i=0; i<n; i++) {
            for(int j=0; j<n; j++) {
                System.out.print(map[i][j]?1:0);
            }
            System.out.println();
        }
        List<Boolean> res = new LinkedList<>();
        for (int[] query : queries) {
            if (map[query[0]][query[1]]) {
                res.add(true);
            } else {
                res.add(false);
            }
        }
        return res;
    }

    public int sumNums(int n) {
        boolean flag = n > 0 && (n += sumNums(n-1)) >0;
        return n;
    }

    public double new21Game(int N, int K, int W) {
        if(K == 0 || K+W-1 <= N) return 1;
        double[] dp = new double[N+1];
        double sum = 1.0;
        double res = 0.0;
        dp[0] = 1.0;
        //有点像是滑动窗口
        for(int i=1; i<=N; i++) {
            dp[i] = sum / W;
            if(i < K)
                sum += dp[i];
            else {
                res += dp[i];
            }
            if (i >= W) {
                sum -= dp[i - W];
            }
        }


        return res;
    }

    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, 1);
        int left = 1, right = 1;
        for(int i=0; i<nums.length-1;i++) {
            res[i] *= left;
            res[nums.length-1-i] *= right;
            left *= nums[i];
            right *= nums[nums.length-1-i];
        }
        res[0] *= right;
        res[nums.length-1] *= left;
        return res;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new LinkedList<>();
        if(matrix.length == 0) {
            return res;
        }
        //int[][] dirx = {{0,1}, {1,0}, {0,-1}, {-1,0}};
        int row = matrix.length-1;
        int col = matrix[0].length-1;
        int temp_row=0, temp_col=0;
        while(res.size() != matrix.length*matrix[0].length) {
            if(temp_row == row && temp_col== col) {
                res.add(matrix[temp_row][temp_col]);
                break;
            }
            circle(matrix, res, temp_row, temp_col, row, col);
            row = Math.max(0, row-1);
            col = Math.max(0, col-1);
            temp_col = Math.min(temp_col+1, matrix[0].length-1);
            temp_row = Math.min(temp_row+1, matrix.length-1);
//            if(temp_row == row && temp_col== col) {
//                res.add(matrix[temp_row][temp_col]);
//            }
        }
        int[] ans = res.stream().mapToInt(Integer::valueOf).toArray();
        return res;
    }

    public void circle(int[][] matrix, List<Integer> res, int s_r, int s_c, int e_r, int e_c) {
        for(int j=s_c; j<=e_c; j++) {
            res.add(matrix[s_r][j]);
        }
        for(int i=s_r+1; i<=e_r; i++) {
            res.add(matrix[i][e_c]);
        }
        for(int j=e_c-1; j>s_c&&e_r>s_r; j--) {
            res.add(matrix[e_r][j]);
        }
        for(int i=e_r; i>s_r&&e_c>s_c; i--) {
            res.add(matrix[i][s_c]);
        }
    }

    public int longestConsecutive(int[] nums) {
        HashSet<Integer> nums_set = new HashSet<>();
        int res = 0;
        int curNum = 0, curLength = 1;
        if(nums.length == 0) {
            return res;
        }
        for(int num : nums) {
            nums_set.add(num);
        }
        for(int num : nums_set) {
            if(!nums_set.contains(num-1)) {
                curLength = 1;
                curNum = num;
                while(nums_set.contains(curNum+1)) {
                    curLength++;
                    curNum++;
                }
                res = Math.max(curLength, res);
            }
        }

        return res;
    }

    public boolean equationsPossible(String[] equations) {
        if(equations.length == 0) {
            return true;
        }
        boolean[][] map = new boolean[26][26];
        LinkedList<String> NotEqual = new LinkedList<>();
        for(String equation : equations) {
            if(equation.charAt(1) == '!') {
                NotEqual.add(equation);
            }
            else
            {
                map[(int) equation.charAt(0) - (int) 'a'][(int) equation.charAt(3) - (int) 'a'] = true;
                map[(int) equation.charAt(3) - (int) 'a'][ (int) equation.charAt(0) - (int) 'a'] = true;
            }
        }
        for(int i=0; i<26; i++) {
            map[i][i] = true;
        }
        for(int i=0; i<26; i++) {
            for(int j=0; j<26; j++) {
                for(int k=0; k<26; k++) {
                    if(map[j][i] && map[i][k])
                        map[j][k] = true;
                }
            }
        }
        for(String equation : NotEqual) {
            if(map[(int) equation.charAt(0) - (int) 'a'][(int) equation.charAt(3) - (int) 'a'])
                return false;
        }
        return true;
    }

    public int translateNum(int num) {
        String src = String.valueOf(num);
        int prev = 1;
        int now = 1;
        int res = 0;
        for(int i=1; i<src.length(); i++) {
            String temp = src.substring(i-1, i+1);
            if(temp.compareTo("25") <= 0 && temp.compareTo("10") >= 0) {
                res = now + prev;
                prev = now;
                now = res;
            }
            else{
            res = now;
            prev = now;
            }
        }
        return res;
    }

    public ListNode reverseBetween(ListNode head, int m, int n) {
        Stack<ListNode> stk = new Stack<>();
        ListNode temp = head;
        ListNode prev = null;   //指向m的节点
        ListNode after = null;
        if(m == 1) {
            for(int i=0; i<n-1; i++) {
                stk.push(temp);
                temp = temp.next;
            }
            head = temp;
            after = temp.next;
            while(!stk.empty()) {
                ListNode curr = stk.pop();
                temp.next = curr;
                temp = curr;
            }
        }
        else {
            for(int i=0; i<n; i++) {
                if(i == m-2) {
                    prev = temp;
                }
                if(i >= m-1) {
                    stk.push(temp);
                }
                temp = temp.next;
            }
            temp = stk.pop();
            after = temp.next;
            prev.next = temp;
            while(!stk.empty()) {
                ListNode curr = stk.pop();
                temp.next = curr;
                temp = curr;
            }
        }
        temp.next = after;
        return head;
    }

    public int[] dailyTemperatures(int[] T) {
        Stack<Pair> stk = new Stack<>();
        int[] res = new int[T.length];
        for(int i=0; i<T.length; i++) {
            if(stk.empty()) {
                stk.push(new Pair(i, T[i]));
            }
            while(!stk.empty() && T[i] > stk.peek().V) {
                Pair temp = stk.pop();
                res[temp.K] = i - temp.K;
            }
            stk.push(new Pair(i, T[i]));
        }
        return res;
    }

    public double[] cutSquares(int[] square1, int[] square2) {
        double[] res = new double[4];
        double[] center1 = {square1[0] + 1.0*square1[2]/2, square1[1] + 1.0*square1[2]/2};
        double[] center2 = {square2[0] + 1.0*square2[2]/2, square2[1] + 1.0*square2[2]/2};
        double[] delta = {center1[0] - center2[0], center1[1] - center2[1]};
        if(delta[0] == 0) {
            double[] y_pos = new double[4];
            y_pos[0] = square1[1];
            y_pos[1] = square1[1] + square1[2];
            y_pos[2] = square2[1];
            y_pos[3] = square2[1] + square2[2];
            Arrays.sort(y_pos);
            res[0] = center1[0];
            res[2] = center1[0];
            res[1] = y_pos[0];
            res[3] = y_pos[3];
        }
        else {
            double k = delta[1] / delta[0];
            double b = center1[1] - k*center1[0];
            double[] x_pos = new double[4];
            if(Math.abs(k) < 1) {
                x_pos[0] = square1[0];
                x_pos[1] = square1[0] + square1[2];
                x_pos[2] = square2[0];
                x_pos[3] = square2[0] + square2[2];
                Arrays.sort(x_pos);
                res[0] = x_pos[0];
                res[1] = res[0]*k + b;
                res[2] = x_pos[3];
                res[3] = res[2]*k + b;
            }
            else {
                x_pos[0] = square1[1];
                x_pos[1] = square1[1] + square1[2];
                x_pos[2] = square2[1];
                x_pos[3] = square2[1] + square2[2];
                Arrays.sort(x_pos);
                res[1] = x_pos[0];
                res[0] = (res[1]-b)/k;
                res[3] = x_pos[3];
                res[2] = (res[3]-b)/k;
                if(res[0] > res[2]) {
                    double temp = res[2];
                    res[2] = res[0];
                    res[0] = temp;
                    temp = res[3];
                    res[3] = res[1];
                    res[1] = temp;
                }
            }

        }
        return res;
    }

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new LinkedList<>();
        if(nums.length < 3) {
            return res;
        }
        for(int first=0; nums[first] <= 0; first++) {
            if(first>0 && nums[first] == nums[first-1])
                continue;
            int second = first+1, third = nums.length-1;
            while(second < third) {
                if(nums[second] + nums[third] < -nums[first]) {
                    while(second<third && nums[second] == nums[++second]){}
                }
                else if (nums[second] + nums[third] > -nums[first]) {
                    while(second<third && nums[third] == nums[--third]){}
                }
                else {
                    List<Integer> temp = new LinkedList<>();
                    temp.add(nums[first]);
                    temp.add(nums[second]);
                    temp.add(nums[third]);
                    res.add(temp);
                    while(second<third && nums[second] == nums[++second]){}
                }
            }
        }

        return res;
    }

    public int[] finalPrices(int[] prices) {
        if(prices.length == 0) {
            return new int[0];
        }
        int[] res = new int[prices.length];
        for(int i=0; i<prices.length; i++) {
            res[i] = prices[i];
            for(int j=i+1; j<prices.length; j++) {
                if(prices[j] <= prices[i]) {
                    res[i] = prices[i] - prices[j];
                    break;
                }
            }
        }
        return res;
    }

    //找两个和为目标值且不重叠的子数组
    public int minSumOfLengths(int[] arr, int target) {
        if(arr.length < 2) {
            return -1;
        }
        int res = 0;
        int flag = 0;
        int[] sum = new int[arr.length];
        sum[0] = arr[0];
        for(int i=1; i<arr.length; i++) {
            sum[i] = sum[i-1] + arr[i];
        }
        for(int i=1; i<=sum.length; i++) {
            if(sum[i-1] == target) {
                res += i;
                flag++;
                int temp = fun(sum, i, target);
                if(temp == -1) {
                    flag--;
                    res -= i;
                    break;
                }
                else{
                    res += temp;
                    flag++;
                    break;
                }
            }
            for(int j=i; j<sum.length; j++) {
                if(sum[j]-sum[j-i] == target) {
                    res += i;
                    flag++;
                    int temp = fun(sum, j+1, target);
                    if(temp == -1) {
                        flag--;
                        res -= i;
                        break;
                    }
                    else{
                        res += temp;
                        flag++;
                        break;
                    }
                }
            }
            if(flag == 2) {
                break;
            }
        }
        if(flag < 2)
            return -1;
        else
            return res;
    }

    public int fun(int[] sum, int left, int target) {
        for(int i=1; i + left <= sum.length; i++) {
            if(sum[left+i-1] - sum[left-1]== target) {
                return i;
            }
            for(int j=left+i; j<sum.length; j++) {
                if(sum[j]-sum[j-i] == target) {
                    return i;
                }
            }
        }
        return -1;
    }

    public int findBestValue(int[] arr, int target) {
        Arrays.sort(arr);
        int sum = 0;
        for(int i=0; i<arr.length; i++) {
            int x = (target-sum)/(arr.length-i); //期待的value
            if(x <= arr[i]) {
                double temp = 1.0*(target - sum)/(arr.length-i);
                if(temp-x>0.5) {
                    return x+1;
                }
                else{
                    return x;
                }
            }
            sum += arr[i];
        }
        return arr[arr.length-1];
    }

    public String longestCommonPrefix(String[] strs) {
        if(strs.length == 0) {
            return new String("");
        }
        int index = -1;
        int min = Integer.MAX_VALUE;
        for(int i=0; i<strs.length; i++) {
            if(strs[i].length()<min) {
                index = i;
                min = strs[i].length();
            }
        }
        StringBuffer sb = new StringBuffer();
        for(int i=0; i<strs[index].length(); i++) {
            boolean flag = true;
            for(int j=0; j<strs.length; j++) {
                if(strs[j].charAt(i) != strs[index].charAt(i)) {
                    flag = false;
                    break;
                }
            }
            if(flag)
                sb.append(strs[index].charAt(i));
            else break;
        }
        return sb.toString();
    }

    public int maxScoreSightseeingPair(int[] A) {
        int res = 0;
        int max = A[0];
        for(int i=1; i<A.length; i++) {
            res = Math.max(max+A[i]-i, res);
            max = Math.max(A[i]+i, max);
        }
        return res;
    }

    public boolean patternMatching(String pattern, String value) {
        if(value.compareTo("") == 0 && pattern.compareTo("") == 0)
            return true;
        if(pattern.compareTo("") == 0) {
            return false;
        }
        if(value.compareTo("") == 0) {
            if(pattern.length()<2)
                return true;
            return false;
        }
        int cntA = 0;
        int cntB = 0;
        for(int i=0; i<pattern.length(); i++) {
            if(pattern.charAt(i) == 'a') {
                cntA++;
            }
            else cntB++;
        }
        if(cntB == 0) {
            if(value.length() % cntA == 0) {
                String A = value.substring(0, value.length() / cntA);
                String merge = merge(pattern, A, "");
                if(merge.compareTo(value) == 0)
                    return true;
            }
            return false;
        }
        if(cntA == 0) {
            if(value.length() % cntB == 0) {
                String B = value.substring(0, value.length() / cntB);
                String merge = merge(pattern, "", B);
                if(merge.compareTo(value) == 0)
                    return true;
            }
            return false;
        }
        for(int LA = 0; LA <= value.length()/cntA; LA++) {
            int LB = 0;
            if((value.length() - LA*cntA) % cntB != 0)
                continue;
            else {
                LB = (value.length() - LA*cntA) / cntB;
                String A = null;
                String B = null;
                boolean aIsDone=false, bIsDone=false;
                for(int i=0; i<=pattern.length(); i++) {
                    if(!aIsDone && pattern.charAt(i) == 'a') {
                        aIsDone = true;
                        if(bIsDone) {
                            A = value.substring(i*LB, i*LB + LA);
                        }
                        else
                            A = value.substring(i*LA, i*LA+LA);
                    }
                    if(!bIsDone && pattern.charAt(i) == 'b') {
                        bIsDone = true;
                        if(aIsDone) {
                            B = value.substring(i*LA, i*LA + LB);
                        }
                        else
                            B = value.substring(i*LB, i*LB+LB);
                    }
                    if(aIsDone && bIsDone) break;
                }
                String merge = merge(pattern, A, B);
                if(merge.compareTo(value) == 0)
                    return true;
            }

        }
        return false;
    }

    public String merge(String pattern, String A, String B) {
        StringBuffer sb = new StringBuffer();
        for(int i=0; i<pattern.length(); i++) {
            if(pattern.charAt(i) == 'a') {
                sb.append(A);
            }
            else sb.append(B);
        }
        return sb.toString();
    }

    public String addBinary(String a, String b) {
        StringBuffer sb = new StringBuffer();
        int i = 0;
        boolean carry = false;
        StringBuffer stra = new StringBuffer(a).reverse();
        StringBuffer strb = new StringBuffer(b).reverse();
        while(i<stra.length() && i<strb.length()) {
            if(stra.charAt(i) == '1' && strb.charAt(i) == '1') {
                if(carry) sb.append('1');
                else    sb.append('0');
                carry = true;
            }
            else if (stra.charAt(i) == '1' || strb.charAt(i) == '1') {
                if(carry) {
                    sb.append('0');
                }
                else{
                    sb.append('1');
                }
            }
            else {
                if(carry) {
                    sb.append('1');
                    carry = false;
                }
                else sb.append('0');
            }
            i++;
        }
        if(i != strb.length()) {
            while(i < strb.length()) {
                if(strb.charAt(i) == '1') {
                    if(carry) sb.append('0');
                    else sb.append('1');
                }
                else {
                    if(carry) sb.append('1');
                    else sb.append('0');
                    carry = false;
                }
                i++;
            }
        }
        else if (i != stra.length()) {
            while(i < stra.length()) {
                if(stra.charAt(i) == '1') {
                    if(carry) sb.append('0');
                    else sb.append('1');
                }
                else {
                    if(carry) sb.append('1');
                    else sb.append('0');
                    carry = false;
                }
                i++;
            }
        }
        if(carry) sb.append('1');
        return sb.reverse().toString();
    }

    public int threeSunClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int min = Integer.MAX_VALUE;
        int ans = 0;
        for(int i=0; i<nums.length; i++) {
            int left = i+1;
            int right = nums.length-1;
            while(left < right) {
                int temp = target - nums[i] - nums[left] - nums[right];
                if(temp == 0) {
                    return target;
                }
                else {
                    if(Math.abs(temp) < min) {
                        min = Math.abs(temp);
                        ans = nums[i] + nums[left] + nums[right];
                    }
                    if(temp > 0) {
                        left++;
                    }
                    else {
                        right--;
                    }
                }
            }
        }
        return ans;
    }

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length()+1];
        dp[0] = true;
        for(int i=1; i<s.length()+1; i++) {
            for(int j=0; j<i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        if(s.length() == 0) {
            return res;
        }
        int left = 0;
        int right = 0;
        HashMap<Character, Boolean> map = new HashMap<>();
        int temp = 0;
        while(right < s.length()) {
            if(map.get(s.charAt(right)) == null || !map.get(s.charAt(right))) {
                map.put(s.charAt(right), true);
                temp++;
                right++;
            }
            else{
                map.put(s.charAt(left), false);
                left++;
                res = Math.max(res, temp);
                temp--;
            }
        }
        return Math.max(res, temp);
    }

    public ListNode removeDuplicateNodes(ListNode head) {
        HashMap<Integer, Boolean> map = new HashMap<>();
        ListNode temp = head;
        ListNode prev = head;
        while(temp != null) {
            if(map.get(temp.val) == null) {
                map.put(temp.val, true);
                prev = temp;
            }
            else {
                prev.next = temp.next;
            }
            temp = temp.next;
        }
        return head;
    }

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for(int i=0; i<n; i++) {
            if(nums[i] <= 0)
                nums[i] = n+1;
        }
        for(int i=0; i<n; i++) {
            int abs = Math.abs(nums[i]);
            if(abs >= 1 && abs <= n && nums[abs-1] > 0)
                nums[abs-1] = -nums[abs-1];
        }
        for(int i=0; i<n; i++) {
            if(nums[i] > 0)
                return i+1;
        }
        return n+1;
    }

    public double average(int[] salary) {
        int maxValue = 0;
        int minValue = Integer.MAX_VALUE;
        int sum = 0;
        for(int i : salary) {
            if(i > maxValue) {
                maxValue = i;
            }
            if(i < minValue) {
                minValue = i;
            }
            sum += i;
        }
        return (1.0*sum-maxValue-minValue)/(salary.length-2);
    }

    public int kthFactor(int n, int k) {
        int cnt = 0;
        for(int i=1; i<=n; i++) {
            if(n % i == 0) {
                cnt++;
            }
            if(cnt == k) {
                return i;
            }
        }
        return -1;
    }

    public int longestSubarray(int[] nums) {
        int res = 0;
        List<Integer> tag = new LinkedList<>();
        for(int i=0; i<nums.length; i++) {
            if(nums[i] == 0) {
                tag.add(i);
            }
        }
        if(tag.size() == nums.length) {
            return 0;
        }
        else if(tag.size() == 0 || tag.size() == 1) {
            return nums.length-1;
        }
        else {
            Iterator<Integer> itr = tag.iterator();
            int prev = itr.next();
            int after = itr.next();
            res = after-1;
            while(itr.hasNext()) {
                int temp = itr.next();
                res = Math.max(res, temp-prev-2);
                prev = after;
                after = temp;
            }
            res = Math.max(res, after-prev-1+nums.length-after-1);
            return res;
        }
    }

    public int minSubArrayLen(int s, int[] nums) {
        int min = Integer.MAX_VALUE;
        int sum = 0;
        int left=0, right=0;
        if(nums.length == 0) {
            return 0;
        }
        else {
            while(right < nums.length) {
                sum += nums[right];
                while(sum >= s) {
                    min = Math.min(min, right-left+1);
                    sum -= nums[left++];
                }
                right++;
            }
        }
        if(min == Integer.MAX_VALUE)
            return 0;
        return min;
    }

    public boolean canReach(int[] arr, int start) {
        boolean[] map = new boolean[arr.length];
        Queue<Integer> queue = new LinkedList<>();
        queue.add(start);
        map[start] = true;
        while(!queue.isEmpty()) {
            int temp = queue.remove();
            if(arr[temp] == 0) {
                return true;
            }
            else {
                if(temp+arr[temp] < arr.length && !map[temp+arr[temp]]) {
                    queue.add(temp+arr[temp]);
                    map[temp+arr[temp]] = true;
                }
                if(temp-arr[temp] >= 0 && !map[temp-arr[temp]]) {
                    queue.add(temp-arr[temp]);
                    map[temp-arr[temp]] = true;
                }
            }
        }
        return false;
    }

    public int findKthLargest(int[] nums, int k) {
        return quickSelect(nums, 0, nums.length - 1, nums.length - k);
    }

    public int quickSelect(int[] a, int l, int r, int index) {
        int q = randomPartition(a, l, r);
        if(q == index) {
            return a[q];
        }else {
            return q < index ? quickSelect(a, q+1, r, index) : quickSelect(a, l, q-1, index);
        }
    }

    public int randomPartition(int[] a, int l, int r) {
        int i = new Random().nextInt(r-l +1) + l;
        swap(a, i, r);
        return partition(a, i, r);
    }

    public int partition(int[] a, int l, int r) {
        int x = a[r], i = l-1;
        for(int j=l; j<r; j++) {
            if(a[j] <= x) {
                swap(a, ++i, j);
            }
        }
        swap(a, i+1, r);
        return i+1;
    }

    public void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if(obstacleGrid.length == 0 || obstacleGrid[0].length == 0
                || obstacleGrid[0][0] == 1) {
            return 0;
        }
        int r = obstacleGrid.length, c = obstacleGrid[0].length;
        int[][] dp = new int[r][c];
        dp[0][0] = 1;
        for(int i=0; i<r; i++) {
            for(int j=0; j<c; j++) {
                if(obstacleGrid[i][j] == 1)
                    continue;
                if(i > 0)
                    dp[i][j] += dp[i-1][j];
                if(j > 0)
                    dp[i][j] += dp[i][j-1];
            }
        }
        return dp[r-1][c-1];
    }

    public int findMin(int[] nums) {
        int left = 0, right = nums.length-1;
        int mid = 0;
        //顺序数组
        if(nums[right] > nums[0]) {
            return nums[0];
        }
        //数组长度小于2时
        while(left < right) {
            mid = (left + right)/2;
            //说明变化点在mid右边
            if(nums[mid] > nums[left]) {
                left = mid;
            }
            //说明变化点在【left, right】区间内
            else {
                if(nums[mid] > nums[mid+1]) {
                    return nums[mid+1];
                }
                //目标点
                if(nums[mid] < nums[mid-1]) {
                    break;
                }
                else {
                    right = mid;
                }
            }
        }
        return nums[mid];
    }
}

public class Main {
    public static void main(String[] args) {
        Solution s = new Solution();
        s.test();
//        int[] test = {0,1,2};
//        int i = 0;
//        System.out.println(test[i] == test[++i]);
//        int i = -9;
//        String s = "-9";
//        System.out.println(Integer.toString(i));
    }
//    @Target(ElementType.TYPE)
//    public @interface MyAnnotaions {
//
//    }
}
