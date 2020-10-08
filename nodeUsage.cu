//#include <iostream>
//
//class Node {
//public:
//    double x;
//    double y;
//    Node* connectedNodes;
//
//    Node(const double x, const double y)
//    {
//        this->x = x;
//        this->y = y;
//    }
//
//    double distance(Node node1, Node node2)
//    {
//        return 0;
//    }
//};
//
//void printNodes(Node* nodes, const int numberOfNodes)
//{
//    printf("\t%p\n", nodes);
//    for (auto i = 0; i < numberOfNodes; i++) {
//        std::cout << "\t"
//            << nodes
//            << i
//            << " ("
//            << nodes[i].x
//            << ", "
//            << nodes[i].y
//            << ")"
//            << std::endl;
//    }
//    std::cout << std::endl;
//}
//
//int main()
//{
//    const auto numNodes = 2;
//    auto n1 = Node(69, 420);
//    auto n2 = Node(8, 740);
//
//    Node nc1[1] = { n2 };
//    Node nc2[1] = { n1 };
//
//    n1.setConnectedNodes(nc1, 1);
//    n2.setConnectedNodes(nc2, 1);
//
//    Node nodes[numNodes] = { n1, n2 };
//    printNodes(nodes, numNodes);
//
//    return 0;
//}