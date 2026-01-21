extends Area3D
class_name Lever

## Lever that can be pulled by the player. Set to activate rising stairs (through signal connection).

@export var deactivated_angle: float = -45.0
@export var activated_angle: float = 45.0
@export var handle_rotation_speed: float = 1.0
@onready var lever_handle: MeshInstance3D = $"lever/lever-box/lever"

signal activated

var _activated: bool
var _activating: bool
var _player_in_area: Player


func _physics_process(delta):
	process_activation(delta)
	if _player_in_area and _player_in_area.use_action_requested:
		activate()


func process_activation(delta):
	if _activating:
		var angle_diff = activated_angle - lever_handle.rotation_degrees.x
		if abs(angle_diff) > 0.1:
			lever_handle.rotation_degrees.x += angle_diff * handle_rotation_speed * delta
		else:
			lever_handle.rotation_degrees.x = activated_angle
			_activating = false
			_activated = true


func activate():
	if not (_activating or _activated):
		_activating = true
		activated.emit()


func deactivate():
	lever_handle.rotation_degrees.x = deactivated_angle
	_activating = false
	_activated = false
	_player_in_area = null


func _on_body_entered(body):
	if body is Player:
		_player_in_area = body


func _on_body_exited(body):
	if body is Player:
		_player_in_area = null


func reset():
	deactivate()
