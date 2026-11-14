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

/// Spawn a background UDP responder for LAN discovery probes.
pub async fn spawn_discovery_responder(cfg: DiscoveryConfig) -> std::io::Result<()> {
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
        logical_cpus: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        auth_required: cfg.auth_required,
    };
    let beacon_bytes = serde_json::to_vec(&beacon).expect("beacon serialization cannot fail");

    info!(
        discovery_port = cfg.port,
        http_port = cfg.http_port,
        node_name = %beacon.name,
        "LAN discovery responder listening"
    );

    tokio::spawn(async move {
        let mut buf = [0_u8; 1024];
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

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn discovery_responder_replies_to_probe() {
        let probe_sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let probe_port = probe_sock.local_addr().unwrap().port();

        // Bind a responder on an ephemeral port using the same logic.
        let resp_sock = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let resp_addr = resp_sock.local_addr().unwrap();

        let beacon = DiscoveryBeacon {
            magic: DiscoveryBeacon::MAGIC.to_string(),
            service: "offload-server".to_string(),
            version: "0.1.0".to_string(),
            name: "test-phone".to_string(),
            http_port: 7878,
            os: "android".to_string(),
            arch: "aarch64".to_string(),
            logical_cpus: 8,
            auth_required: false,
        };
        let beacon_bytes = serde_json::to_vec(&beacon).unwrap();

        tokio::spawn(async move {
            let mut buf = [0_u8; 512];
            if let Ok((len, peer)) = resp_sock.recv_from(&mut buf).await
                && let Ok(probe) = serde_json::from_slice::<DiscoveryProbe>(&buf[..len])
                && probe.magic == DiscoveryProbe::MAGIC
            {
                let _ = resp_sock.send_to(&beacon_bytes, peer).await;
            }
        });

        let probe_bytes = serde_json::to_vec(&DiscoveryProbe::new()).unwrap();
        probe_sock.send_to(&probe_bytes, resp_addr).await.unwrap();

        let mut recv_buf = [0_u8; 1024];
        let (len, from) = tokio::time::timeout(
            Duration::from_secs(2),
            probe_sock.recv_from(&mut recv_buf),
        )
        .await
        .unwrap()
        .unwrap();

        assert_eq!(from.port(), resp_addr.port());
        assert_ne!(probe_port, 0);
        let decoded: DiscoveryBeacon = serde_json::from_slice(&recv_buf[..len]).unwrap();
        assert_eq!(decoded.name, "test-phone");
        assert_eq!(decoded.http_port, 7878);
    }
}
