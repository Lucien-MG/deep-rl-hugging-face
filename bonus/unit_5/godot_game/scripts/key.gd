extends Area3D
class_name Key

## A key that can be collected by the player. Used to unlock the treasure chest.

@onready var collision_shape := $CollisionShape3D

signal collected

## Positions that the key can spawn in, relative to the parent node
@export var possible_positions = [
	Vector3(-0.329, 0.1, -2.643),
	Vector3(-3.592, 0.1, -2.643)
]
# Randomize the initial spawn position to create more varied env copies
var current_position_index: int = randi_range(0, possible_positions.size())


func _on_body_entered(body):
	var player = body as Player
	if player:
		collect()


func collect():
	collected.emit()
	visible = false
	collision_shape.set_deferred("disabled", true)


func reset():
	visible = true
	collision_shape.set_deferred("disabled", false)
	# Switch the position on each restart to the next possible position
	current_position_index = (current_position_index + 1) % possible_positions.size()
	position = possible_positions[current_position_index]
