### 240113_화물운송

```cpp
#define OPTION 3
// OPTION == 1: 변형된 다익스트라 (visited 미사용)
// OPTION == 2: 변형된 다익스트라 (visited 사용)
// OPTION == 3: 이진탐색 + BFS

#include <vector>
#include <queue>
using namespace std;

const int MAX_N = 1000;
const int MAX_K = 4000;
const int MAX_WEIGHT = 30000;

// 간선(도로) 정보를 저장하는 구조체
struct Road {
    int to, weight; // { 도착 도시, 최대 중량 }
    bool operator<(const Road& other) const { return weight < other.weight; }
};

// 그래프 구조
vector<Road> graph[MAX_N];
bool visited[MAX_N];
int dist[MAX_N];    // 최대 중량 저장
int N;              // 노드(도시)의 개수
int K;              // 간선(도로)의 개수


//////////////////////////////////////////////////////////////////////
void add(int sCity, int eCity, int mLimit);

void init(int N, int K, int sCity[], int eCity[], int mLimit[]) {
    ::N = N;
    ::K = K;
    for (int i = 0; i < N; i++) graph[i].clear();
    for (int i = 0; i < K; i++) add(sCity[i], eCity[i], mLimit[i]);
}

void add(int sCity, int eCity, int mLimit) {
    graph[sCity].push_back({ eCity, mLimit });
}

#if OPTION == 1
// 최대 중량 경로 찾기 (변형된 다익스트라 알고리즘) - visited 미사용
int calculate(int sCity, int eCity) {
    if (sCity == eCity) return -1;

    for (int i = 0; i < N; i++) dist[i] = -1;
    priority_queue<Road> pq;

    dist[sCity] = MAX_WEIGHT;
    pq.push({ sCity, dist[sCity] });

    while (!pq.empty()) {
        auto curr = pq.top(); pq.pop();

        if (curr.to == eCity) return curr.weight;
        if (curr.weight < dist[curr.to]) continue;

        for (const auto& next: graph[curr.to]) {
            int next_weight = min(curr.weight, next.weight);

            if (next_weight > dist[next.to]) {
                dist[next.to] = next_weight;
                pq.push({ next.to, dist[next.to] });
            }
        }
    }
    return -1;
}
#elif OPTION == 2
// 최대 중량 경로 찾기 (변형된 다익스트라 알고리즘) - visited 사용
int calculate(int sCity, int eCity) {
    if (sCity == eCity) return -1;

    for (int i = 0; i < N; i++) {
        visited[i] = false;
        dist[i] = -1;
    }
    priority_queue<Road> pq;
    dist[sCity] = MAX_WEIGHT;
    pq.push({ sCity, dist[sCity] });

    while (!pq.empty()) {
        auto curr = pq.top(); pq.pop();

        if (visited[curr.to]) continue;
        visited[curr.to] = true;

        if (curr.to == eCity) return curr.weight;

        for (const auto& next: graph[curr.to]) {
            int next_weight = min(curr.weight, next.weight);

            if (!visited[next.to] && next_weight > dist[next.to]) {
                dist[next.to] = next_weight;
                pq.push({ next.to, dist[next.to] });
            }
        }
    }
    return -1;
}
#elif OPTION == 3
// 주어진 중량으로 목적지에 도달할 수 있는지 확인하는 BFS
bool canReach(int sCity, int eCity, int weight) {
    for (int i = 0; i < N; i++) visited[i] = false;
    queue<Road> q;

    q.push({ sCity, 0 });
    visited[sCity] = true;

    while (!q.empty()) {
        auto curr = q.front(); q.pop();

        if (curr.to == eCity) return true;

        for (const auto& next : graph[curr.to]) {
            // 해당 도로의 최대 중량이 원하는 중량보다 크거나 같고, 방문하지 않은 도시라면 방문
            if (!visited[next.to] && next.weight >= weight) {
                visited[next.to] = true;
                q.push(next);
            }
        }
    }
    return false;
}

int calculate(int sCity, int eCity) {
    if (sCity == eCity) return -1;

    // 이진 탐색을 위한 경계값 설정
    int low = 1;            // 최소 가능한 중량
    int high = MAX_WEIGHT;  // 최대 가능한 중량
    int result = -1;        // 결과값 (도달 불가능할 경우 -1)

    // 이진 탐색
    while (low <= high) {
        int mid = low + (high - low) / 2; // 중간값 (현재 확인할 중량)

        // 해당 중량으로 목적지에 도달할 수 있는지 확인
        // 도달 가능하면 결과 갱신하고 더 큰 중량 확인
        if (canReach(sCity, eCity, mid)) { result = mid; low = mid + 1; }
        // 도달 불가능하면 더 작은 중량 확인
        else { high = mid - 1; }
    }
    return result;
}
#endif
```


### 241102_최소차이경로

```cpp
#include <vector>
#include <queue>
#include <unordered_map>
using namespace std;

const int MAX_N = 1000;	// 노드(도시)의 최대 개수
const int MAX_K = 10000;	// 간선(도로)의 최대 개수

struct Edge {
	int to, mCost;
};
vector<Edge> graph[MAX_N];

enum State { ADDED, REMOVED };
struct Road {
	int sCity, eCity, mCost;
	State state;
};
Road roads[MAX_K];
int roadCnt;
unordered_map<int, int> roadMap;

int N;
int K;

//////////////////////////////////////////////////////////////////////
void add(int mId, int sCity, int eCity, int mCost);

void init(int N, int K, int mId[], int sCity[], int eCity[], int mCost[]) {
	::N = N;
	::K = K;

	roadMap.clear();
	roadCnt = 0;
	for (int i = 0; i < N; i++) graph[i].clear();
	for (int i = 0; i < K; i++) add(mId[i], sCity[i], eCity[i], mCost[i]);
}

void add(int mId, int sCity, int eCity, int mCost) {
	roads[roadCnt++] = { sCity, eCity, mCost, ADDED };
	graph[sCity].push_back({ eCity, mCost });
}

void remove(int mId) {
	int rIdx = roadMap[mId];
	roads[rIdx].state = REMOVED;
}


int cost(int sCity, int eCity) {

	return 0;

}
```
