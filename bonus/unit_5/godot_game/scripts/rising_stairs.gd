extends AnimatableBody3D
class_name RisingStairs

## A rising stairs platform, connected to be activated when the player pulls a lever through signals.

@export var raise_speed: float = 1.0
@export var raise_height: float = 1.0
@onready var initial_transform = transform

var _activating: bool


func _physics_process(delta):
	process_activation(delta)


func process_activation(delta):
	if _activating:
		var target_y: float = initial_transform.origin.y + raise_height
		if position.y < target_y:
			position += Vector3.UP * raise_speed * delta
		else:
			position.y = target_y
			_activating = false


func activate():
	_activating = true


func deactivate():
	transform = initial_transform
	_activating = false


func _on_lever_activated():
	activate()


func reset():
	deactivate()
