import * as THREE from "three";
import { cellToWorld, FLIGHT_HEIGHT } from "./scene.js";

const BODY_COLOR = 0xf59e0b;

export class Drone {
  constructor(scene, startCell) {
    this.group = new THREE.Group();
    this.model = new THREE.Group();
    this.model.scale.setScalar(1.3);
    this.group.add(this.model);
    this.rotors = [];

    this.bodyMaterial = new THREE.MeshStandardMaterial({
      color: BODY_COLOR, roughness: 0.4, emissive: 0x000000,
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.12, 0.35), this.bodyMaterial);
    body.castShadow = true;
    this.model.add(body);

    // Small nose marker so the flying direction is visible.
    const nose = new THREE.Mesh(
      new THREE.BoxGeometry(0.12, 0.08, 0.1),
      new THREE.MeshStandardMaterial({ color: 0x111827 })
    );
    nose.position.set(0, 0, 0.2);
    this.model.add(nose);

    const armMat = new THREE.MeshStandardMaterial({ color: 0x1f2937 });
    for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.03, 0.03), armMat);
      arm.position.set(sx * 0.2, 0, sz * 0.2);
      arm.rotation.y = sx * sz * Math.PI / 4;
      this.model.add(arm);

      const rotor = new THREE.Mesh(
        new THREE.BoxGeometry(0.34, 0.012, 0.04),
        new THREE.MeshStandardMaterial({ color: 0xe5e7eb })
      );
      rotor.position.set(sx * 0.3, 0.06, sz * 0.3);
      this.model.add(rotor);
      this.rotors.push(rotor);
    }

    // Blood package hangs under the drone while it is carried.
    this.cargo = new THREE.Mesh(
      new THREE.BoxGeometry(0.22, 0.16, 0.22),
      new THREE.MeshStandardMaterial({ color: 0xdc2626 })
    );
    this.cargo.position.y = -0.18;
    this.cargo.visible = false;
    this.model.add(this.cargo);

    // Blue ring that shows the drone is waiting (hovering).
    this.waitRing = new THREE.Mesh(
      new THREE.RingGeometry(0.38, 0.45, 32),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0, side: THREE.DoubleSide })
    );
    this.waitRing.rotation.x = -Math.PI / 2;
    this.waitRing.position.y = -0.05;
    this.model.add(this.waitRing);

    this.flashColor = new THREE.Color(BODY_COLOR);
    this.flashTime = 0;
    this.flashDuration = 1;
    this.shakeTime = 0;
    this.bumpTime = 0;
    this.bumpDir = new THREE.Vector2(0, 0);
    this.waiting = false;

    // heading is the angle around the vertical axis; forward = (sin, 0, cos).
    this.heading = Math.atan2(1, 1);
    this.targetHeading = this.heading;

    this.target = new THREE.Vector3();
    this.setCell(startCell[0], startCell[1]);
    this.group.position.copy(this.target);
    scene.add(this.group);
  }

  setCell(x, y) {
    this.target.copy(cellToWorld(x, y, FLIGHT_HEIGHT));
  }

  snapToTarget() {
    this.group.position.copy(this.target);
  }

  faceDirection(dx, dz) {
    if (dx !== 0 || dz !== 0) this.targetHeading = Math.atan2(dx, dz);
  }

  setCargoVisible(visible) {
    this.cargo.visible = visible;
  }

  setWaiting(on) {
    this.waiting = on;
  }

  // Colored flash: red = penalty, green = reward, etc.
  flash(hex, seconds = 0.5) {
    this.flashColor.setHex(hex);
    this.flashTime = seconds;
    this.flashDuration = seconds;
  }

  // Short shake, used for collisions and invalid actions.
  shake(seconds = 0.45) {
    this.shakeTime = seconds;
  }

  // Lunge toward a tower and bounce back.
  bump(dx, dz) {
    this.bumpTime = 0.6;
    this.bumpDir.set(dx, dz);
  }

  update(dt, time) {
    this.group.position.lerp(this.target, 1 - Math.exp(-dt * 8));

    // Turn smoothly toward the flying direction.
    let diff = this.targetHeading - this.heading;
    diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    this.heading += diff * Math.min(dt * 8, 1);
    this.group.rotation.y = this.heading;

    this.model.position.y = Math.sin(time * (this.waiting ? 2 : 3)) * (this.waiting ? 0.04 : 0.02);

    if (this.bumpTime > 0) {
      this.bumpTime -= dt;
      const t = 1 - Math.max(this.bumpTime, 0) / 0.6;
      const push = Math.sin(Math.PI * t) * 0.72;
      this.group.position.x += this.bumpDir.x * push;
      this.group.position.z += this.bumpDir.y * push;
    }

    if (this.shakeTime > 0) {
      this.shakeTime -= dt;
      this.group.position.x += Math.sin(time * 90) * 0.03;
      this.group.position.z += Math.cos(time * 80) * 0.03;
    }

    if (this.flashTime > 0) {
      this.flashTime -= dt;
      const k = Math.max(this.flashTime / this.flashDuration, 0);
      this.bodyMaterial.emissive.copy(this.flashColor).multiplyScalar(k * 0.9);
    } else {
      this.bodyMaterial.emissive.setHex(0x000000);
    }

    const ringTarget = this.waiting ? 0.55 + 0.35 * Math.sin(time * 6) : 0;
    this.waitRing.material.opacity += (ringTarget - this.waitRing.material.opacity) * Math.min(dt * 10, 1);
    this.waitRing.scale.setScalar(1 + (this.waiting ? 0.12 * Math.sin(time * 6) : 0));

    const spin = this.waiting ? 28 : 40;
    for (const r of this.rotors) r.rotation.y += dt * spin;
  }
}
