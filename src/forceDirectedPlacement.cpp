#include "forceDirectedPlacement.hpp"
#include "vector2.hpp"

const static double REPULSION = 3.0;
const static double ATTRACTION = 3.0;
const static int SPRING_LENGTH = 100;
const static int MAX_ITERATIONS = 5;
const static double DAMPING = 0.5;

/// <summary>
/// calcs the repulsion force between 2 nodes
/// </summary>
/// <param name="x">The node that the force is acting on.</param>
/// <param name="y">The node creating the force.</param>
/// <returns>A vector representing the repulsion force.</returns>
inline Vector2 repulsionForce(Node& node1, Node& node2)
{
	auto proximity = std::max(node1.distance(node2), 1.0);
	double force = -(REPULSION / (proximity * proximity));
	double angle = node1.position->getBearingAngle(*node2.position);
	return Vector2(0.0, 0.0);
}

/// <summary>
/// calculates the attriction force between 2 nodes
/// </summary>
/// <param name="node1">the node that the force is acting on</param>
/// <param name="node2">the node that is creating the force</param>
/// <param name="lengthOfSpring">the length of the spring</param>
/// <returns>a vector representing the attraction force</returns>
inline Vector2 calcAttractionForce(Node& node1, Node& node2, double lengthOfSpring) {
	auto proximity = std::max((int) node1.distance(node2), 1);
	double force = ATTRACTION * std::max(proximity - lengthOfSpring, 0.0);
	double angle = node1.position->getBearingAngle(*node2.position);
	return Vector2(force, angle);
}

double totalDisplacement = 0.0;
int stopCount = 0;

void arrangeNodes(std::vector<Node>& nodes)
{
	totalDisplacement = 0.0;

	for (int i = 0; i < MAX_ITERATIONS; ++i)
	{
		update(nodes);
	}
}

/// <summary>
/// force directed placement algorithm
/// </summary>
/// <param name="nodes">Nodes you want to run the alg on</param>
/// <param name="damping">rate at which spring goes back to normal</param>
/// <param name="springLength">length of spring</param>
/// <param name="maxIterations">number of iterations you want
/// the alg to run. Higher = More accuracy</param>
void update(std::vector<Node>& nodes, double damping, int springLength, int maxIterations)
{
	Vector2 zero = (Vector2)Vector2::ZERO;
	for (auto& node : nodes)
	{
		auto netForce = zero;
		auto currentPosition = Vector2(node.position->distance(zero), node.position->getBearingAngle(zero));
		// determine repulsion between nodes
		for (auto& otherNode : nodes) {
			if (otherNode.id == node.id) continue;
			auto f = repulsionForce(node, otherNode);
			netForce.x += f.x;
			netForce.y += f.y;
		}

		// determine attraction caused by connections
		for (auto& child : node.connectedNodes) {
			auto temp = calcAttractionForce(node, *child, springLength);
			netForce += temp;
		}

		/*for (auto& parent : nodes) {
			if (parent.connectedNodes.begin(), parent.connectedNodes.end(), ) continue;
			auto f = calcAttractionForce(node, parent, springLength);
			netForce.x += f.x;
			netForce.y += f.y;
		}*/

		// apply net force to node velocity
		auto& velocity = *node.velocity;
		velocity = ((velocity + netForce) * (damping));
		node.velocity = &velocity;

		// apply velocity to node position
		auto& position = *node.position;
		position = velocity + currentPosition;
		node.position = &position;

		
	}
}

/// <summary>
/// Runs update with default values
/// </summary>
/// <param name="nodes">nodes you want to run the alg on</param>
void update(std::vector<Node>& nodes)
{
	update(nodes, DAMPING, SPRING_LENGTH, MAX_ITERATIONS);
}