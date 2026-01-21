extends Node3D
class_name LevelManager

## Nodes that can be reset, e.g. lever and others
var resetables: Array[Node]


func _ready() -> void:
	var nodes = find_children("*")
	for node in nodes:
		if node.is_in_group("resetable"):
			resetables.append(node)


func reset_all_resetables():
	for node in resetables:
		assert(
			node.has_method("reset"),
			"Resetable group node %s does not implement reset() method." % node.get_path()
		)
		node.reset()
