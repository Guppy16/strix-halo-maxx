module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    %t = aie.tile(0, 5)
    aie.packet_flow(2) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%t, DMA : 0>
    } {keep_pkt_header = false}
    aie.packet_flow(4) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%m, DMA : 0>
      aie.packet_dest<%m, DMA : 5>
    } {keep_pkt_header = false}
    aie.packet_flow(1) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%m, DMA : 0>
    } {keep_pkt_header = false}
    aie.packet_flow(3) {
      aie.packet_source<%s, DMA : 0>
      aie.packet_dest<%m, DMA : 5>
    } {keep_pkt_header = false}
  }
}
