//! UDP responder that answers LAN discovery probes with a [`DiscoveryBeacon`].

use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

use offload_protocol::{DISCOVERY_MULTICAST_ADDR, DiscoveryBeacon, DiscoveryProbe};
use tokio::net::UdpSocket;
use tracing::{debug, info, warn};

/// Metadata advertised by the discovery responder.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub port: u16,
    pub http_port: u16,
    pub node_name: String,
    pub auth_required: bool,
}

/// Probes are tiny JSON objects (~45 bytes); anything larger is not ours and is ignored.
const PROBE_BUF_BYTES: usize = 1024;

/// Binds `cfg.port` (0 = ephemeral) and spawns a task answering discovery probes.
///
/// Returns the bound UDP port.
///
/// # Errors
///
/// Returns the I/O error if the UDP socket cannot be bound.
pub async fn spawn_discovery_responder(cfg: DiscoveryConfig) -> std::io::Result<u16> {
    let bind_addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, cfg.port));
    let socket = UdpSocket::bind(bind_addr).await?;
    let _ = socket.set_broadcast(true);

    if let Ok(mcast_ip) = DISCOVERY_MULTICAST_ADDR.parse::<Ipv4Addr>()
        && let Err(err) = socket.join_multicast_v4(mcast_ip, Ipv4Addr::UNSPECIFIED)
    {
        warn!(error = %err, "could not join discovery multicast group; broadcast still active");
    }

    let beacon = DiscoveryBeacon {
        magic: DiscoveryBeacon::MAGIC.to_string(),
        service: "offload-server".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        name: cfg.node_name,
        http_port: cfg.http_port,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
        auth_required: cfg.auth_required,
    };
    let beacon_bytes = serde_json::to_vec(&beacon).expect("beacon serialization cannot fail");
    let bound_port = socket.local_addr()?.port();

    info!(
        discovery_port = bound_port,
        http_port = cfg.http_port,
        node_name = %beacon.name,
        "LAN discovery responder listening"
    );

    tokio::spawn(async move {
        let mut buf = [0_u8; PROBE_BUF_BYTES];
        loop {
            match socket.recv_from(&mut buf).await {
                Ok((len, peer)) => {
                    if let Ok(probe) = serde_json::from_slice::<DiscoveryProbe>(&buf[..len])
                        && probe.magic == DiscoveryProbe::MAGIC
                    {
                        debug!(%peer, "responding to discovery probe");
                        let _ = socket.send_to(&beacon_bytes, peer).await;
                    }
                }
                Err(err) => {
                    warn!(error = %err, "discovery socket recv error");
                }
            }
        }
    });

    Ok(bound_port)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn discovery_responder_replies_to_probe() {
        let port = spawn_discovery_responder(DiscoveryConfig {
            port: 0,
            http_port: 7878,
            node_name: "test-phone".to_string(),
            auth_required: true,
        })
        .await
        .unwrap();

        let probe_sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let probe_bytes = serde_json::to_vec(&DiscoveryProbe::new()).unwrap();
        // Junk first: the responder must ignore it and keep serving.
        probe_sock
            .send_to(b"not json", ("127.0.0.1", port))
            .await
            .unwrap();
        probe_sock
            .send_to(&probe_bytes, ("127.0.0.1", port))
            .await
            .unwrap();

        let mut recv_buf = [0_u8; 1024];
        let (len, _) =
            tokio::time::timeout(Duration::from_secs(2), probe_sock.recv_from(&mut recv_buf))
                .await
                .expect("responder must reply to a valid probe")
                .unwrap();

        let decoded: DiscoveryBeacon = serde_json::from_slice(&recv_buf[..len]).unwrap();
        assert_eq!(decoded.magic, DiscoveryBeacon::MAGIC);
        assert_eq!(decoded.name, "test-phone");
        assert_eq!(decoded.http_port, 7878);
        assert!(decoded.auth_required);
    }
}
