struct Obstacle {
    position: vec2<f32>,
    radius: f32
};

struct Uniforms {
    resolution: vec2<f32>,
    _pad0: f32,
    _pad1: f32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> obstacles: array<Obstacle>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) localPos: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    var out: VertexOutput;
    
    let obs: Obstacle = obstacles[instanceIndex];
    
    if (obs.radius <= 0.0) {
        out.position = vec4<f32>(-10.0, -10.0, 0.0, 1.0);
        return out;
    }
    
    var corner: vec2<f32>;
    switch (vertexIndex) {
        case 0u: { corner = vec2<f32>(-1.0, -1.0); }
        case 1u: { corner = vec2<f32>(1.0, -1.0); }
        case 2u: { corner = vec2<f32>(-1.0, 1.0); }
        case 3u: { corner = vec2<f32>(1.0, -1.0); }
        case 4u: { corner = vec2<f32>(1.0, 1.0); }
        default: { corner = vec2<f32>(-1.0, 1.0); }
    }
    
    let aspect: f32 = uniforms.resolution.x / uniforms.resolution.y;
    let screenPos: vec2<f32> = obs.position * 2.0 - 1.0;
    let scale: vec2<f32> = vec2<f32>(obs.radius / aspect, obs.radius);
    
    out.position = vec4<f32>(screenPos + corner * scale, 0.0, 1.0);
    out.localPos = corner;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist: f32 = length(in.localPos);
    
    if (dist > 1.0) {
        discard;
    }
    
    let edge: f32 = smoothstep(1.0, 0.9, dist);
    let inner: f32 = smoothstep(0.9, 0.0, dist);
    
    let edgeColor: vec3<f32> = vec3<f32>(0.9, 0.3, 0.3);
    let innerColor: vec3<f32> = vec3<f32>(0.3, 0.3, 0.4);
    
    let color: vec3<f32> = mix(innerColor, edgeColor, 1.0 - inner);
    
    return vec4<f32>(color, edge);
}
