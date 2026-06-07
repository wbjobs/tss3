struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    density: f32,
    pressure: f32,
    color: vec3<f32>,
    _pad: f32
};

struct Uniforms {
    resolution: vec2<f32>,
    particleRadius: f32,
    _pad: f32
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) localPos: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) particlePos: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertexIndex: u32,
    @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
    var out: VertexOutput;
    
    let p: Particle = particles[instanceIndex];
    
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
    let radius: f32 = uniforms.particleRadius;
    
    let screenPos: vec2<f32> = p.position * 2.0 - 1.0;
    let scale: vec2<f32> = vec2<f32>(radius / aspect, radius);
    
    out.position = vec4<f32>(screenPos + corner * scale, 0.0, 1.0);
    out.localPos = corner;
    out.color = p.color;
    out.particlePos = p.position;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dist: f32 = length(in.localPos);
    
    if (dist > 1.0) {
        discard;
    }
    
    let alpha: f32 = smoothstep(1.0, 0.0, dist);
    let glow: f32 = smoothstep(0.3, 0.0, dist) * 0.5 + 0.5;
    
    var color: vec3<f32> = in.color * glow;
    
    let highlight: f32 = smoothstep(0.7, 0.2, dist);
    color = color + highlight * 0.3;
    
    return vec4<f32>(color, alpha * 0.9);
}
